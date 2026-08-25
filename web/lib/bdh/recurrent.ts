/**
 * Recurrent (state-space) inference for the BDH toy.
 *
 * This is the form that makes BDH's claim tangible: generation carries a
 * fixed-size state rho (one d-vector per neuron per layer) instead of a KV
 * cache that grows with the sequence. Reading a token touches rho; writing a
 * token adds a single rank-1 outer product to it.
 *
 * It is mathematically the same computation as the parallel, triangular-matrix
 * form in `bdh.py` (and in ./parallel.ts) — scripts/check-equivalence.ts
 * asserts that the two agree numerically.
 */

import {
  addOuter,
  countPositive,
  layerNorm,
  matvecAT,
  mulberry32,
} from "./linalg";
import { ropeApply, ropeFreqs, type ToyConfig, type ToyParams } from "./model";

export interface LayerTrace {
  /** paper x_{t,l}: positive neuron activations, (Dx v)^+ */
  x: Float32Array;
  /** paper (Dy LN(rho x))^+ : the multiplicative gate coming out of attention */
  gate: Float32Array;
  /** paper y_{t,l} = gate (elementwise) x — the vector the paper reports as ~5% sparse at scale */
  y: Float32Array;
  xPositive: number;
  gatePositive: number;
  yPositive: number;
}

export interface StepTrace {
  position: number;
  token: number;
  layers: LayerTrace[];
  logits: Float32Array;
}

/** Per-layer synaptic state. rho[l] is n x d: row i is neuron i's own d-vector. */
export interface RhoState {
  rho: Float32Array[];
  position: number;
}

export function createRhoState(c: ToyConfig): RhoState {
  return {
    rho: Array.from({ length: c.layers }, () => new Float32Array(c.n * c.d)),
    position: 0,
  };
}

/** Total scalars held in working memory during inference, independent of length. */
export function rhoFloats(c: ToyConfig): number {
  return c.layers * c.n * c.d;
}

/** What a Transformer would instead cache after `tokens` tokens, at the same d and depth. */
export function kvCacheFloats(c: ToyConfig, tokens: number): number {
  return 2 * c.layers * tokens * c.d;
}

interface Scratch {
  x: Float32Array;
  xNext: Float32Array;
  xIn: Float32Array;
  xs: Float32Array;
  qr: Float32Array;
  a: Float32Array;
  an: Float32Array;
  gate: Float32Array;
  y: Float32Array;
  ym: Float32Array;
  yn: Float32Array;
  sum: Float32Array;
  logits: Float32Array;
}

function makeScratch(c: ToyConfig): Scratch {
  return {
    x: new Float32Array(c.d),
    xNext: new Float32Array(c.d),
    xIn: new Float32Array(c.d),
    xs: new Float32Array(c.n),
    qr: new Float32Array(c.n),
    a: new Float32Array(c.d),
    an: new Float32Array(c.d),
    gate: new Float32Array(c.n),
    y: new Float32Array(c.n),
    ym: new Float32Array(c.d),
    yn: new Float32Array(c.d),
    sum: new Float32Array(c.d),
    logits: new Float32Array(c.vocab),
  };
}

/**
 * One BDH-GPU token step, mirroring the body of `BDH.forward` in bdh.py.
 * Returns the next-token logits, plus per-layer activations when asked.
 */
export function stepToken(
  params: ToyParams,
  c: ToyConfig,
  freqs: Float32Array,
  state: RhoState,
  token: number,
  scratch: Scratch,
  wantTrace: boolean,
): StepTrace {
  const { d, n } = c;
  const s = scratch;
  const pos = state.position;

  // x = LN(embed[token])
  layerNorm(params.embed, token * d, d, s.x, 0);

  const layers: LayerTrace[] = [];

  for (let l = 0; l < c.layers; l++) {
    s.xIn.set(s.x);

    // x_sparse = ReLU(x @ Dx): lift the residual into neuron space, keep positives
    matvecAT(params.Dx, d, n, s.x, s.qr); // reuse qr as the pre-activation buffer
    for (let i = 0; i < n; i++) s.xs[i] = s.qr[i] > 0 ? s.qr[i] : 0;

    // Q = K = RoPE(x_sparse, pos): attention scores are inner products of
    // positive neuron vectors, rotated by position.
    ropeApply(s.xs, 0, n, freqs, pos, s.qr, 0);

    // Read working memory: a = rho^T q  ==  sum over tau < pos of <q_pos, k_tau> v_tau
    matvecAT(state.rho[l], n, d, s.qr, s.a);
    layerNorm(s.a, 0, d, s.an, 0);

    // gate = ReLU(LN(a) @ Dy), then y = gate * x_sparse
    matvecAT(params.Dy, d, n, s.an, s.y); // reuse y as the pre-activation buffer
    for (let i = 0; i < n; i++) s.gate[i] = s.y[i] > 0 ? s.y[i] : 0;
    for (let i = 0; i < n; i++) s.y[i] = s.gate[i] * s.xs[i];

    // Back down to d through E, then the residual update
    matvecAT(params.E, n, d, s.y, s.ym);
    layerNorm(s.ym, 0, d, s.yn, 0);
    for (let i = 0; i < d; i++) s.sum[i] = s.xIn[i] + s.yn[i];
    layerNorm(s.sum, 0, d, s.xNext, 0);

    // Write working memory: rho += k (outer) v, with v the layer input.
    // Only neurons that fired have a non-zero row to write.
    addOuter(state.rho[l], s.qr, n, s.xIn, d);

    if (wantTrace) {
      layers.push({
        x: s.xs.slice(),
        gate: s.gate.slice(),
        y: s.y.slice(),
        xPositive: countPositive(s.xs, 0, n),
        gatePositive: countPositive(s.gate, 0, n),
        yPositive: countPositive(s.y, 0, n),
      });
    }

    s.x.set(s.xNext);
  }

  matvecAT(params.head, d, c.vocab, s.x, s.logits);
  state.position += 1;

  return {
    position: pos,
    token,
    layers,
    logits: wantTrace ? s.logits.slice() : s.logits,
  };
}

export interface SampleOptions {
  temperature: number;
  topK: number;
}

export function sampleFromLogits(
  logits: Float32Array,
  { temperature, topK }: SampleOptions,
  rand: () => number,
): number {
  const v = logits.length;
  const scaled = new Float32Array(v);
  const t = Math.max(temperature, 1e-3);
  for (let i = 0; i < v; i++) scaled[i] = logits[i] / t;

  let threshold = -Infinity;
  if (topK > 0 && topK < v) {
    const order = Array.from({ length: v }, (_, i) => i).sort(
      (a, b) => scaled[b] - scaled[a],
    );
    threshold = scaled[order[topK - 1]];
  }

  let max = -Infinity;
  for (let i = 0; i < v; i++) {
    if (scaled[i] >= threshold && scaled[i] > max) max = scaled[i];
  }
  let total = 0;
  const probs = new Float32Array(v);
  for (let i = 0; i < v; i++) {
    if (scaled[i] < threshold) continue;
    const p = Math.exp(scaled[i] - max);
    probs[i] = p;
    total += p;
  }
  let r = rand() * total;
  for (let i = 0; i < v; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  for (let i = v - 1; i >= 0; i--) if (probs[i] > 0) return i;
  return 0;
}

/**
 * A stateful generation session: feed the prompt one byte at a time, then keep
 * sampling. Each call is O(n*d*L) work and adds nothing to memory.
 */
export class ToySession {
  readonly config: ToyConfig;
  private readonly params: ToyParams;
  private readonly freqs: Float32Array;
  private readonly scratch: Scratch;
  private readonly state: RhoState;
  private rand: () => number;
  lastTrace: StepTrace | null = null;

  constructor(params: ToyParams, config: ToyConfig, seed = 1337) {
    this.params = params;
    this.config = config;
    this.freqs = ropeFreqs(config.n, config.ropeTheta);
    this.scratch = makeScratch(config);
    this.state = createRhoState(config);
    this.rand = mulberry32(seed);
  }

  get position(): number {
    return this.state.position;
  }

  feed(token: number, wantTrace = false): StepTrace {
    const trace = stepToken(
      this.params,
      this.config,
      this.freqs,
      this.state,
      token,
      this.scratch,
      wantTrace,
    );
    if (wantTrace) this.lastTrace = trace;
    return trace;
  }

  /** Fraction of rho entries that have ever been written to, per layer. */
  rhoOccupancy(): number[] {
    return this.state.rho.map((m) => {
      let nz = 0;
      for (let i = 0; i < m.length; i++) if (m[i] !== 0) nz++;
      return nz / m.length;
    });
  }

  sample(logits: Float32Array, options: SampleOptions): number {
    return sampleFromLogits(logits, options, this.rand);
  }
}

/** Byte-level tokenisation, exactly what train.py does to the corpus. */
export function encodeBytes(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

/**
 * Render bytes for display the way train.py's `errors="backslashreplace"` does:
 * printable ASCII as-is, everything else escaped so nothing is silently hidden.
 */
export function decodeBytesForDisplay(bytes: Uint8Array | number[]): string {
  let out = "";
  for (const b of bytes) {
    if (b === 10) out += "\n";
    else if (b === 9) out += "\t";
    else if (b >= 32 && b < 127) out += String.fromCharCode(b);
    else out += `\\x${b.toString(16).padStart(2, "0")}`;
  }
  return out;
}
