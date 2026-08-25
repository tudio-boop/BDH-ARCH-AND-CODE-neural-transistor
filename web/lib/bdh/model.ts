/**
 * A tiny BDH-GPU(n, d) model, ported by hand from the reference PyTorch
 * implementation in this repo (`bdh.py`) with nh = 1 head and dropout = 0.
 *
 * Naming follows the paper (arXiv:2509.26507, Eq. 8) rather than the PyTorch
 * file, because the paper's names are the ones people read about. The mapping:
 *
 *   paper            bdh.py               here
 *   ---------------  -------------------  -------------------------------
 *   Dx  (d -> n)     self.encoder         params.Dx      (stored d x n)
 *   Dy  (d -> n)     self.encoder_v       params.Dy      (stored d x n)
 *   E   (n -> d)     self.decoder         params.E       (stored n x d)
 *   x_t (in R^n)     x_sparse             the sparse, positive neuron vector
 *   LN(E y) in R^d   x (the residual)     the low-rank "synaptic" vector
 *   rho (n x d)      implicit in `scores` explicit state, see recurrent.ts
 *   U                the RoPE frequencies rope()
 *
 * Note the direction of the names: what `bdh.py` calls `encoder` lifts the
 * d-dimensional residual into the n-dimensional neuron space, which is the
 * paper's *decoder* Dx. The tensors are identical; only the labels differ.
 */

import { gaussian, mulberry32 } from "./linalg";

export interface ToyConfig {
  /** low-rank / synaptic dimension (paper: d) */
  d: number;
  /** neuron count (paper: n) */
  n: number;
  /** layers, all sharing the same parameters (paper: L) */
  layers: number;
  /** byte-level vocabulary, exactly as train.py treats the corpus */
  vocab: number;
  /** RoPE base, matching bdh.py's `theta=2**16` */
  ropeTheta: number;
}

export const TOY_CONFIG: ToyConfig = {
  d: 32,
  n: 256,
  layers: 4,
  vocab: 256,
  ropeTheta: 65536,
};

export interface ToyParams {
  /** vocab x d */
  embed: Float32Array;
  /** d x n — lifts the residual into neuron space (bdh.py: encoder) */
  Dx: Float32Array;
  /** d x n — lifts the attention read-out into neuron space (bdh.py: encoder_v) */
  Dy: Float32Array;
  /** n x d — projects the neuron space back down (bdh.py: decoder) */
  E: Float32Array;
  /** d x vocab */
  head: Float32Array;
}

export const PARAM_KEYS = ["embed", "Dx", "Dy", "E", "head"] as const;
export type ParamKey = (typeof PARAM_KEYS)[number];

export function paramShapes(c: ToyConfig): Record<ParamKey, [number, number]> {
  return {
    embed: [c.vocab, c.d],
    Dx: [c.d, c.n],
    Dy: [c.d, c.n],
    E: [c.n, c.d],
    head: [c.d, c.vocab],
  };
}

export function paramCount(c: ToyConfig): number {
  return Object.values(paramShapes(c)).reduce((t, [a, b]) => t + a * b, 0);
}

/** bdh.py initialises every parameter matrix with normal(std=0.02). */
export function initParams(c: ToyConfig, seed = 1337): ToyParams {
  const rand = mulberry32(seed);
  const make = (size: number) => {
    const a = new Float32Array(size);
    for (let i = 0; i < size; i++) a[i] = gaussian(rand) * 0.02;
    return a;
  };
  const shapes = paramShapes(c);
  return {
    embed: make(shapes.embed[0] * shapes.embed[1]),
    Dx: make(shapes.Dx[0] * shapes.Dx[1]),
    Dy: make(shapes.Dy[0] * shapes.Dy[1]),
    E: make(shapes.E[0] * shapes.E[1]),
    head: make(shapes.head[0] * shapes.head[1]),
  };
}

/**
 * bdh.py:get_freqs — pairs of neuron dimensions share a frequency, so RoPE
 * rotates the neuron space two coordinates at a time. This is the paper's
 * decay/rotation operator U folded into the query/key phases.
 */
export function ropeFreqs(n: number, theta: number): Float32Array {
  const f = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const q = Math.floor(j / 2) * 2;
    f[j] = 1 / theta ** (q / n) / (2 * Math.PI);
  }
  return f;
}

/**
 * Rotate a neuron-space vector by its position phases (bdh.py:Attention.rope).
 * `sign = -1` applies the transpose, which is what the backward pass needs.
 */
export function ropeApply(
  src: Float32Array,
  srcOff: number,
  n: number,
  freqs: Float32Array,
  pos: number,
  dst: Float32Array,
  dstOff: number,
  sign = 1,
): void {
  for (let j = 0; j < n; j += 2) {
    let phase = (pos * freqs[j]) % 1;
    if (phase < 0) phase += 1;
    phase *= 2 * Math.PI;
    const c = Math.cos(phase);
    const s = Math.sin(phase) * sign;
    const a = src[srcOff + j];
    const b = src[srcOff + j + 1];
    dst[dstOff + j] = a * c - b * s;
    dst[dstOff + j + 1] = b * c + a * s;
  }
}
