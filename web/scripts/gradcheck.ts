/**
 * Finite-difference check of the hand-written backward pass in lib/bdh/train.ts.
 *
 *   npm run toy:gradcheck
 */

import { mulberry32 } from "../lib/bdh/linalg";
import {
  initParams,
  PARAM_KEYS,
  ropeFreqs,
  type ParamKey,
  type ToyConfig,
} from "../lib/bdh/model";
import { allocCache, crossEntropy, forward } from "../lib/bdh/parallel";
import { allocBackward, backward, zeroGrads } from "../lib/bdh/train";

const config: ToyConfig = { d: 6, n: 8, layers: 2, vocab: 11, ropeTheta: 65536 };
const T = 5;
const EPS = 1e-3;
const TOLERANCE = 2e-2;

const rand = mulberry32(7);
const params = initParams(config, 99);
const freqs = ropeFreqs(config.n, config.ropeTheta);
const tokens = new Uint8Array(T);
const targets = new Uint8Array(T);
for (let t = 0; t < T; t++) {
  tokens[t] = Math.floor(rand() * config.vocab);
  targets[t] = Math.floor(rand() * config.vocab);
}

const cache = allocCache(config, T);
const scratch = allocBackward(config, T);

/**
 * ReLU makes the loss only piecewise differentiable, so a finite difference is
 * meaningless whenever the perturbation switches a unit on or off. Fingerprint
 * the activation pattern so those coordinates can be skipped instead of counted
 * as gradient errors.
 */
function activationPattern(): string {
  const bits: number[] = [];
  for (const L of cache.layers) {
    let acc = 0;
    let count = 0;
    for (let i = 0; i < T * config.n; i++) {
      acc = (acc * 2 + (L.xs[i] > 0 ? 1 : 0)) % 1000000007;
      acc = (acc * 2 + (L.gate[i] > 0 ? 1 : 0)) % 1000000007;
      count++;
    }
    bits.push(acc, count);
  }
  return bits.join(":");
}
const grads = {
  embed: new Float32Array(params.embed.length),
  Dx: new Float32Array(params.Dx.length),
  Dy: new Float32Array(params.Dy.length),
  E: new Float32Array(params.E.length),
  head: new Float32Array(params.head.length),
};
const dLogits = new Float32Array(T * config.vocab);

function lossOf(): number {
  forward(params, config, freqs, tokens, cache);
  return crossEntropy(cache.logits, T, config.vocab, targets, null);
}

forward(params, config, freqs, tokens, cache);
const baseLoss = crossEntropy(cache.logits, T, config.vocab, targets, dLogits);
zeroGrads(grads);
backward(params, config, freqs, cache, dLogits, grads, scratch);

console.log(`base loss ${baseLoss.toFixed(6)} (uniform would be ${Math.log(config.vocab).toFixed(6)})`);

const basePattern = activationPattern();

let worst = 0;
let worstLabel = "";
let checked = 0;
let skipped = 0;

for (const key of PARAM_KEYS as readonly ParamKey[]) {
  const arr = params[key];
  const attempts = Math.min(40, arr.length);
  let keyWorst = 0;
  let keyChecked = 0;
  for (let s = 0; s < attempts; s++) {
    const i = Math.floor(rand() * arr.length);
    const original = arr[i];
    arr[i] = original + EPS;
    const up = lossOf();
    const upPattern = activationPattern();
    arr[i] = original - EPS;
    const down = lossOf();
    const downPattern = activationPattern();
    arr[i] = original;
    if (upPattern !== basePattern || downPattern !== basePattern) {
      skipped++;
      continue;
    }
    const numeric = (up - down) / (2 * EPS);
    const analytic = grads[key][i];
    const scale = Math.max(1e-4, Math.abs(numeric) + Math.abs(analytic));
    const rel = Math.abs(numeric - analytic) / scale;
    checked++;
    keyChecked++;
    if (rel > keyWorst) keyWorst = rel;
    if (rel > worst) {
      worst = rel;
      worstLabel = `${key}[${i}] numeric=${numeric.toExponential(4)} analytic=${analytic.toExponential(4)}`;
    }
  }
  console.log(
    `  ${key.padEnd(6)} ${String(keyChecked).padStart(2)} coords, worst relative error ${keyWorst.toExponential(3)}`,
  );
}

console.log(
  `\nchecked ${checked} coordinates (${skipped} skipped: perturbation crossed a ReLU kink), worst ${worst.toExponential(3)}`,
);
console.log(`worst case: ${worstLabel}`);

if (!(worst < TOLERANCE)) {
  console.error(`FAIL: worst relative error ${worst} exceeds ${TOLERANCE}`);
  process.exit(1);
}
console.log("PASS");
