/**
 * Backward pass and AdamW for the BDH toy, written out by hand.
 *
 * Only scripts/train-toy.ts and scripts/gradcheck.ts import this; the deployed
 * pages never do, so none of it ships to the browser. The gradients are checked
 * against finite differences in scripts/gradcheck.ts.
 */

import {
  layerNormBackward,
  matmul,
  matmulAT,
  matmulBT,
} from "./linalg";
import {
  PARAM_KEYS,
  ropeApply,
  type ParamKey,
  type ToyConfig,
  type ToyParams,
} from "./model";
import type { ForwardCache } from "./parallel";

export interface BackwardScratch {
  dx: Float32Array;
  dxIn: Float32Array;
  dsum: Float32Array;
  dym: Float32Array;
  dy: Float32Array;
  dgate: Float32Array;
  dxs: Float32Array;
  dpreN: Float32Array;
  dan: Float32Array;
  da: Float32Array;
  dscores: Float32Array;
  dsym: Float32Array;
  dqr: Float32Array;
  tmpN: Float32Array;
  tmpD: Float32Array;
}

export function allocBackward(c: ToyConfig, T: number): BackwardScratch {
  const { d, n } = c;
  return {
    dx: new Float32Array(T * d),
    dxIn: new Float32Array(T * d),
    dsum: new Float32Array(T * d),
    dym: new Float32Array(T * d),
    dy: new Float32Array(T * n),
    dgate: new Float32Array(T * n),
    dxs: new Float32Array(T * n),
    dpreN: new Float32Array(T * n),
    dan: new Float32Array(T * d),
    da: new Float32Array(T * d),
    dscores: new Float32Array(T * T),
    dsym: new Float32Array(T * T),
    dqr: new Float32Array(T * n),
    tmpN: new Float32Array(T * n),
    tmpD: new Float32Array(d),
  };
}

/** Accumulates gradients into `grads`; `dLogits` comes from crossEntropy(). */
export function backward(
  params: ToyParams,
  c: ToyConfig,
  freqs: Float32Array,
  cache: ForwardCache,
  dLogits: Float32Array,
  grads: ToyParams,
  s: BackwardScratch,
): void {
  const { d, n, vocab } = c;
  const T = cache.T;
  const last = cache.layers[c.layers - 1];
  const xFinal = c.layers > 0 ? last.xOut : cache.x0;

  // logits = x @ head
  matmulAT(xFinal, T, d, dLogits, vocab, grads.head, true);
  matmulBT(dLogits, T, vocab, params.head, d, s.dx);

  for (let l = c.layers - 1; l >= 0; l--) {
    const L = cache.layers[l];

    // x_out = LN(x_in + y_norm)
    for (let t = 0; t < T; t++) {
      layerNormBackward(L.xOut, t * d, L.sumInvStd[t], s.dx, t * d, d, s.dsum, t * d);
    }
    // y_norm = LN(y_mlp)
    for (let t = 0; t < T; t++) {
      layerNormBackward(L.yn, t * d, L.ymInvStd[t], s.dsum, t * d, d, s.dym, t * d);
    }
    // the residual sends the same gradient down both paths
    s.dxIn.set(s.dsum.subarray(0, T * d));

    // y_mlp = y @ E
    matmulAT(L.y, T, n, s.dym, d, grads.E, true);
    matmulBT(s.dym, T, d, params.E, n, s.dy);

    // y = gate * x_sparse
    for (let i = 0; i < T * n; i++) {
      s.dgate[i] = s.dy[i] * L.xs[i];
      s.dxs[i] = s.dy[i] * L.gate[i];
    }

    // gate = ReLU(LN(a) @ Dy)
    for (let i = 0; i < T * n; i++) s.dpreN[i] = L.gate[i] > 0 ? s.dgate[i] : 0;
    matmulAT(L.an, T, d, s.dpreN, n, grads.Dy, true);
    matmulBT(s.dpreN, T, n, params.Dy, d, s.dan);

    // a_norm = LN(a)
    for (let t = 0; t < T; t++) {
      layerNormBackward(L.an, t * d, L.aInvStd[t], s.dan, t * d, d, s.da, t * d);
    }

    // a = scores @ x_in  (x_in is also the attention value)
    matmulBT(s.da, T, d, L.xIn, T, s.dscores);
    for (let i = 0; i < T; i++) {
      for (let j = i; j < T; j++) s.dscores[i * T + j] = 0;
    }
    matmulAT(L.scores, T, T, s.da, d, s.dxIn, true);

    // scores = tril(Q K^T, -1) with Q = K
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        s.dsym[i * T + j] = s.dscores[i * T + j] + s.dscores[j * T + i];
      }
    }
    matmul(s.dsym, T, T, L.qr, n, s.dqr);

    // Q = RoPE(x_sparse): the adjoint is the same rotation, reversed
    for (let t = 0; t < T; t++) {
      ropeApply(s.dqr, t * n, n, freqs, t, s.tmpN, t * n, -1);
    }
    for (let i = 0; i < T * n; i++) s.dxs[i] += s.tmpN[i];

    // x_sparse = ReLU(x_in @ Dx)
    for (let i = 0; i < T * n; i++) s.dpreN[i] = L.xs[i] > 0 ? s.dxs[i] : 0;
    matmulAT(L.xIn, T, d, s.dpreN, n, grads.Dx, true);
    matmulBT(s.dpreN, T, n, params.Dx, d, s.dxIn, true);

    s.dx.set(s.dxIn.subarray(0, T * d));
  }

  // x_0 = LN(embed[token])
  for (let t = 0; t < T; t++) {
    layerNormBackward(
      cache.x0,
      t * d,
      cache.embedInvStd[t],
      s.dx,
      t * d,
      d,
      s.tmpD,
      0,
    );
    const row = cache.tokens[t] * d;
    for (let i = 0; i < d; i++) grads.embed[row + i] += s.tmpD[i];
  }
}

export interface AdamWState {
  m: ToyParams;
  v: ToyParams;
  step: number;
}

export function createAdamW(p: ToyParams): AdamWState {
  const like = (): ToyParams => ({
    embed: new Float32Array(p.embed.length),
    Dx: new Float32Array(p.Dx.length),
    Dy: new Float32Array(p.Dy.length),
    E: new Float32Array(p.E.length),
    head: new Float32Array(p.head.length),
  });
  return { m: like(), v: like(), step: 0 };
}

export interface AdamWOptions {
  lr: number;
  weightDecay: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export const ADAMW_DEFAULTS: AdamWOptions = {
  lr: 1e-3,
  weightDecay: 0.1,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
};

export function adamwStep(
  params: ToyParams,
  grads: ToyParams,
  state: AdamWState,
  o: AdamWOptions,
): void {
  state.step += 1;
  const bc1 = 1 - o.beta1 ** state.step;
  const bc2 = 1 - o.beta2 ** state.step;
  for (const key of PARAM_KEYS as readonly ParamKey[]) {
    const p = params[key];
    const g = grads[key];
    const m = state.m[key];
    const v = state.v[key];
    for (let i = 0; i < p.length; i++) {
      const gi = g[i];
      m[i] = o.beta1 * m[i] + (1 - o.beta1) * gi;
      v[i] = o.beta2 * v[i] + (1 - o.beta2) * gi * gi;
      const mh = m[i] / bc1;
      const vh = v[i] / bc2;
      p[i] -= o.lr * (mh / (Math.sqrt(vh) + o.eps) + o.weightDecay * p[i]);
    }
  }
}

export function zeroGrads(g: ToyParams): void {
  for (const key of PARAM_KEYS as readonly ParamKey[]) g[key].fill(0);
}

export function gradNorm(g: ToyParams): number {
  let total = 0;
  for (const key of PARAM_KEYS as readonly ParamKey[]) {
    const a = g[key];
    for (let i = 0; i < a.length; i++) total += a[i] * a[i];
  }
  return Math.sqrt(total);
}

export function clipGrads(g: ToyParams, maxNorm: number): number {
  const norm = gradNorm(g);
  if (norm > maxNorm) {
    const scale = maxNorm / (norm + 1e-6);
    for (const key of PARAM_KEYS as readonly ParamKey[]) {
      const a = g[key];
      for (let i = 0; i < a.length; i++) a[i] *= scale;
    }
  }
  return norm;
}
