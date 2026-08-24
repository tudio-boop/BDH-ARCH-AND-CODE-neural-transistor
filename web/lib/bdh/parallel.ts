/**
 * Token-parallel forward pass, a line-for-line port of `BDH.forward` in bdh.py
 * (with nh = 1 and dropout = 0).
 *
 * The browser demo does not need this — it uses the recurrent form in
 * ./recurrent.ts. It exists because it is the form you can differentiate
 * cheaply, so scripts/train-toy.ts trains with it, and because comparing the
 * two forms is how we check the recurrent implementation is right.
 */

import {
  invStdOf,
  layerNorm,
  matmul,
  matmulAT,
  scoresStrictLowerTri,
} from "./linalg";
import { ropeApply, type ToyConfig, type ToyParams } from "./model";

export interface LayerCache {
  xIn: Float32Array;
  xs: Float32Array;
  qr: Float32Array;
  scores: Float32Array;
  a: Float32Array;
  an: Float32Array;
  aInvStd: Float32Array;
  gate: Float32Array;
  y: Float32Array;
  ym: Float32Array;
  yn: Float32Array;
  ymInvStd: Float32Array;
  sum: Float32Array;
  xOut: Float32Array;
  sumInvStd: Float32Array;
}

export interface ForwardCache {
  T: number;
  tokens: Uint8Array;
  x0: Float32Array;
  embedInvStd: Float32Array;
  layers: LayerCache[];
  logits: Float32Array;
  scratchN: Float32Array;
}

export function allocCache(c: ToyConfig, T: number): ForwardCache {
  const { d, n } = c;
  return {
    T,
    tokens: new Uint8Array(T),
    x0: new Float32Array(T * d),
    embedInvStd: new Float32Array(T),
    layers: Array.from({ length: c.layers }, () => ({
      xIn: new Float32Array(T * d),
      xs: new Float32Array(T * n),
      qr: new Float32Array(T * n),
      scores: new Float32Array(T * T),
      a: new Float32Array(T * d),
      an: new Float32Array(T * d),
      aInvStd: new Float32Array(T),
      gate: new Float32Array(T * n),
      y: new Float32Array(T * n),
      ym: new Float32Array(T * d),
      yn: new Float32Array(T * d),
      ymInvStd: new Float32Array(T),
      sum: new Float32Array(T * d),
      xOut: new Float32Array(T * d),
      sumInvStd: new Float32Array(T),
    })),
    logits: new Float32Array(T * c.vocab),
    scratchN: new Float32Array(T * n),
  };
}

export function forward(
  params: ToyParams,
  c: ToyConfig,
  freqs: Float32Array,
  tokens: Uint8Array,
  cache: ForwardCache,
): Float32Array {
  const { d, n } = c;
  const T = tokens.length;
  cache.tokens.set(tokens);

  for (let t = 0; t < T; t++) {
    cache.embedInvStd[t] = invStdOf(params.embed, tokens[t] * d, d);
    layerNorm(params.embed, tokens[t] * d, d, cache.x0, t * d);
  }

  let x = cache.x0;

  for (let l = 0; l < c.layers; l++) {
    const L = cache.layers[l];
    L.xIn.set(x.subarray(0, T * d));

    // x_sparse = ReLU(x @ Dx)
    matmul(L.xIn, T, d, params.Dx, n, cache.scratchN);
    for (let i = 0; i < T * n; i++) {
      const v = cache.scratchN[i];
      L.xs[i] = v > 0 ? v : 0;
    }

    // Q = K = RoPE(x_sparse) at absolute positions; the inner product below
    // then only depends on the distance between tokens.
    for (let t = 0; t < T; t++) {
      ropeApply(L.xs, t * n, n, freqs, t, L.qr, t * n);
    }

    // scores = tril(Q K^T, -1); a = scores @ x
    scoresStrictLowerTri(L.qr, T, n, L.scores);
    matmul(L.scores, T, T, L.xIn, d, L.a);

    for (let t = 0; t < T; t++) {
      L.aInvStd[t] = invStdOf(L.a, t * d, d);
      layerNorm(L.a, t * d, d, L.an, t * d);
    }

    // gate = ReLU(LN(a) @ Dy); y = gate * x_sparse
    matmul(L.an, T, d, params.Dy, n, cache.scratchN);
    for (let i = 0; i < T * n; i++) {
      const v = cache.scratchN[i];
      L.gate[i] = v > 0 ? v : 0;
      L.y[i] = L.gate[i] * L.xs[i];
    }

    // back down through E, LN, residual, LN
    matmul(L.y, T, n, params.E, d, L.ym);
    for (let t = 0; t < T; t++) {
      L.ymInvStd[t] = invStdOf(L.ym, t * d, d);
      layerNorm(L.ym, t * d, d, L.yn, t * d);
    }
    for (let i = 0; i < T * d; i++) L.sum[i] = L.xIn[i] + L.yn[i];
    for (let t = 0; t < T; t++) {
      L.sumInvStd[t] = invStdOf(L.sum, t * d, d);
      layerNorm(L.sum, t * d, d, L.xOut, t * d);
    }

    x = L.xOut;
  }

  matmul(x, T, d, params.head, c.vocab, cache.logits);
  return cache.logits;
}

/** Mean cross-entropy over the sequence, and dLogits in `dLogits`. */
export function crossEntropy(
  logits: Float32Array,
  T: number,
  vocab: number,
  targets: Uint8Array,
  dLogits: Float32Array | null,
): number {
  let loss = 0;
  for (let t = 0; t < T; t++) {
    const off = t * vocab;
    let max = -Infinity;
    for (let i = 0; i < vocab; i++) if (logits[off + i] > max) max = logits[off + i];
    let sum = 0;
    for (let i = 0; i < vocab; i++) sum += Math.exp(logits[off + i] - max);
    const logSum = Math.log(sum) + max;
    loss += logSum - logits[off + targets[t]];
    if (dLogits) {
      for (let i = 0; i < vocab; i++) {
        const p = Math.exp(logits[off + i] - logSum);
        dLogits[off + i] = (p - (i === targets[t] ? 1 : 0)) / T;
      }
    }
  }
  return loss / T;
}

/** Fraction of positive entries in the y vectors (paper: ~5% for trained BDH-GPU). */
export function activationSparsity(
  cache: ForwardCache,
  c: ToyConfig,
  T: number,
): { x: number; gate: number; y: number } {
  let xPos = 0;
  let gatePos = 0;
  let yPos = 0;
  const total = c.layers * T * c.n;
  for (const L of cache.layers) {
    for (let i = 0; i < T * c.n; i++) {
      if (L.xs[i] > 0) xPos++;
      if (L.gate[i] > 0) gatePos++;
      if (L.y[i] > 0) yPos++;
    }
  }
  return { x: xPos / total, gate: gatePos / total, y: yPos / total };
}

export { matmulAT };
