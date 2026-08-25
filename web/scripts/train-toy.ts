/**
 * Trains the tiny BDH-GPU whose weights ship with the web demo.
 *
 *   npm run toy:train -- --steps 2500 --batch 12 --block 64
 *
 * Same corpus as the repo's train.py (byte-level tiny Shakespeare), same
 * optimiser settings (AdamW, lr 1e-3, weight decay 0.1) — just ~617x fewer
 * parameters, so that a forward pass fits in a browser tab. Writes
 * public/bdh-toy-weights.json.
 *
 * Nothing in the deployed site depends on running this: the weights it produces
 * are committed.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { mulberry32 } from "../lib/bdh/linalg";
import {
  initParams,
  paramCount,
  paramShapes,
  PARAM_KEYS,
  ropeFreqs,
  TOY_CONFIG,
  type ParamKey,
} from "../lib/bdh/model";
import {
  activationSparsity,
  allocCache,
  crossEntropy,
  forward,
} from "../lib/bdh/parallel";
import { decodeBytesForDisplay, ToySession } from "../lib/bdh/recurrent";
import {
  ADAMW_DEFAULTS,
  adamwStep,
  allocBackward,
  backward,
  clipGrads,
  createAdamW,
  zeroGrads,
} from "../lib/bdh/train";
import { quantizeParams, WEIGHTS_FORMAT } from "../lib/bdh/weights";

const here = dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = join(here, ".cache");
const CORPUS_PATH = join(CACHE_DIR, "input.txt");
const CORPUS_URL =
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

function arg(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
}

const STEPS = arg("steps", 4000);
const BATCH = arg("batch", 10);
const BLOCK = arg("block", 128);
const LR = arg("lr", ADAMW_DEFAULTS.lr);
const SEED = arg("seed", 1337);
const LOG_EVERY = arg("log", 50);
const SAMPLE_EVERY = arg("sample", 500);
const WARMUP = arg("warmup", 100);
const config = TOY_CONFIG;

/** Linear warmup then cosine decay to 10% — the toy gets one short run, so it helps. */
function lrAt(step: number): number {
  if (step < WARMUP) return (LR * (step + 1)) / WARMUP;
  const progress = (step - WARMUP) / Math.max(1, STEPS - WARMUP);
  return LR * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress)));
}

async function loadCorpus(): Promise<Uint8Array> {
  mkdirSync(CACHE_DIR, { recursive: true });
  if (!existsSync(CORPUS_PATH)) {
    process.stdout.write(`downloading corpus from ${CORPUS_URL}\n`);
    const res = await fetch(CORPUS_URL);
    if (!res.ok) throw new Error(`corpus download failed: ${res.status}`);
    writeFileSync(CORPUS_PATH, Buffer.from(await res.arrayBuffer()));
  }
  return new Uint8Array(readFileSync(CORPUS_PATH));
}

const params = initParams(config, SEED);
const grads = {
  embed: new Float32Array(params.embed.length),
  Dx: new Float32Array(params.Dx.length),
  Dy: new Float32Array(params.Dy.length),
  E: new Float32Array(params.E.length),
  head: new Float32Array(params.head.length),
};
const freqs = ropeFreqs(config.n, config.ropeTheta);
const cache = allocCache(config, BLOCK);
const scratch = allocBackward(config, BLOCK);
const dLogits = new Float32Array(BLOCK * config.vocab);
const adam = createAdamW(params);
const rand = mulberry32(SEED);

const x = new Uint8Array(BLOCK);
const y = new Uint8Array(BLOCK);

type Bytes = Uint8Array<ArrayBufferLike>;

let trainData: Bytes = new Uint8Array(0);
let valData: Bytes = new Uint8Array(0);

function sampleBatchInto(data: Bytes): void {
  const start = Math.floor(rand() * (data.length - BLOCK - 1));
  for (let i = 0; i < BLOCK; i++) {
    x[i] = data[start + i];
    y[i] = data[start + i + 1];
  }
}

/**
 * Held-out loss on a fixed set of windows. The RNG is re-seeded per call so
 * every evaluation sees exactly the same bytes — otherwise comparing float32
 * against int8-packed weights would just be measuring sampling noise.
 */
function evaluate(data: Bytes, batches: number): number {
  const pick = mulberry32(0xe7a1);
  let total = 0;
  for (let b = 0; b < batches; b++) {
    const start = Math.floor(pick() * (data.length - BLOCK - 1));
    for (let i = 0; i < BLOCK; i++) {
      x[i] = data[start + i];
      y[i] = data[start + i + 1];
    }
    forward(params, config, freqs, x, cache);
    total += crossEntropy(cache.logits, BLOCK, config.vocab, y, null);
  }
  return total / batches;
}

function generateSample(prompt: string, tokens: number): string {
  const session = new ToySession(params, config, 20250824);
  const promptBytes = new TextEncoder().encode(prompt);
  let logits = session.feed(promptBytes[0]).logits;
  for (let i = 1; i < promptBytes.length; i++) {
    logits = session.feed(promptBytes[i]).logits;
  }
  const out: number[] = [];
  for (let i = 0; i < tokens; i++) {
    const next = session.sample(logits, { temperature: 1, topK: 3 });
    out.push(next);
    logits = session.feed(next).logits;
  }
  return prompt + decodeBytesForDisplay(out);
}

async function main(): Promise<void> {
const corpus = await loadCorpus();
const splitAt = Math.floor(corpus.length * 0.9);
trainData = corpus.subarray(0, splitAt);
valData = corpus.subarray(splitAt);

console.log(
  `BDH-GPU(n=${config.n}, d=${config.d}), L=${config.layers}, byte vocab ${config.vocab}`,
);
console.log(
  `${paramCount(config).toLocaleString()} parameters | corpus ${corpus.length.toLocaleString()} bytes | ` +
    `block ${BLOCK} batch ${BATCH} steps ${STEPS} lr ${LR}`,
);

const startedAt = Date.now();
let ema = 0;
let emaSteps = 0;
const history: { step: number; loss: number }[] = [];
let firstLoss = 0;

for (let step = 0; step < STEPS; step++) {
  zeroGrads(grads);
  let stepLoss = 0;
  for (let b = 0; b < BATCH; b++) {
    sampleBatchInto(trainData);
    forward(params, config, freqs, x, cache);
    stepLoss += crossEntropy(cache.logits, BLOCK, config.vocab, y, dLogits);
    for (let i = 0; i < dLogits.length; i++) dLogits[i] /= BATCH;
    backward(params, config, freqs, cache, dLogits, grads, scratch);
  }
  stepLoss /= BATCH;
  if (step === 0) firstLoss = stepLoss;
  ema += stepLoss;
  emaSteps += 1;
  clipGrads(grads, 1.0);
  adamwStep(params, grads, adam, { ...ADAMW_DEFAULTS, lr: lrAt(step) });

  if (step % LOG_EVERY === 0 || step === STEPS - 1) {
    const mean = ema / emaSteps;
    history.push({ step, loss: Number(mean.toFixed(4)) });
    const elapsed = (Date.now() - startedAt) / 1000;
    const rate = (step + 1) / elapsed;
    console.log(
      `step ${String(step).padStart(5)}/${STEPS}  loss ${mean.toFixed(4)}  ` +
        `${rate.toFixed(2)} steps/s  eta ${(((STEPS - step) / rate) / 60).toFixed(1)}m`,
    );
    ema = 0;
    emaSteps = 0;
  }
  if (SAMPLE_EVERY > 0 && step > 0 && step % SAMPLE_EVERY === 0) {
    console.log(`  sample: ${JSON.stringify(generateSample("To be or ", 80))}`);
  }
}

const trainLoss = evaluate(trainData, 40);
const valLoss = evaluate(valData, 40);
sampleBatchInto(valData);
forward(params, config, freqs, x, cache);
const sparsity = activationSparsity(cache, config, BLOCK);

console.log(`\ntrain loss ${trainLoss.toFixed(4)} | val loss ${valLoss.toFixed(4)}`);
console.log(
  `positive activations: x ${(sparsity.x * 100).toFixed(1)}% | ` +
    `gate ${(sparsity.gate * 100).toFixed(1)}% | y ${(sparsity.y * 100).toFixed(1)}%`,
);
console.log(`sample: ${JSON.stringify(generateSample("To be or ", 120))}`);

const quantized = quantizeParams(params, config);

// What the int8 packing costs, on the same held-out windows as above. The
// packed weights are the ones the browser actually runs.
const exact = { ...params };
for (const key of PARAM_KEYS as readonly ParamKey[]) {
  params[key] = quantized.dequantized[key];
}
const valLossQuantized = evaluate(valData, 40);
for (const key of PARAM_KEYS as readonly ParamKey[]) params[key] = exact[key];

// Keep the unpacked weights around so the losses can be re-measured without
// paying for another training run.
writeFileSync(
  join(CACHE_DIR, "params-float32.json"),
  JSON.stringify(
    Object.fromEntries(
      (PARAM_KEYS as readonly ParamKey[]).map((key) => [
        key,
        Array.from(exact[key]),
      ]),
    ),
  ),
);

console.log(
  `val loss after int8 packing ${valLossQuantized.toFixed(4)} ` +
    `(delta ${(valLossQuantized - valLoss >= 0 ? "+" : "") + (valLossQuantized - valLoss).toFixed(4)})`,
);

const outPath = join(here, "..", "public", "bdh-toy-weights.json");
const payload = {
  format: WEIGHTS_FORMAT,
  note:
    "Tiny BDH-GPU trained by web/scripts/train-toy.ts on byte-level tiny Shakespeare. " +
    "Weights are int8 with a scale per row. Not a paper-scale model.",
  config,
  shapes: paramShapes(config),
  training: {
    parameters: paramCount(config),
    steps: STEPS,
    batch: BATCH,
    block: BLOCK,
    lr: LR,
    optimizer: "AdamW",
    schedule: `linear warmup ${WARMUP} steps, then cosine decay to 10%`,
    weightDecay: ADAMW_DEFAULTS.weightDecay,
    firstLoss: Number(firstLoss.toFixed(4)),
    trainLoss: Number(trainLoss.toFixed(4)),
    valLoss: Number(valLoss.toFixed(4)),
    valLossInt8: Number(valLossQuantized.toFixed(4)),
    history,
    corpusBytes: corpus.length,
    positiveActivations: {
      x: Number(sparsity.x.toFixed(4)),
      gate: Number(sparsity.gate.toFixed(4)),
      y: Number(sparsity.y.toFixed(4)),
    },
    seconds: Math.round((Date.now() - startedAt) / 1000),
  },
  tensors: quantized.tensors,
};
writeFileSync(outPath, JSON.stringify(payload));
console.log(
  `\nwrote ${outPath} (${(JSON.stringify(payload).length / 1024).toFixed(0)} KB)`,
);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
