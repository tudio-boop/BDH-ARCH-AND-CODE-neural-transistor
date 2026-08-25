/**
 * Minimal dense linear algebra on Float32Array, row-major.
 *
 * Everything the BDH toy needs, with no dependencies: the whole point of this
 * demo is that a BDH-GPU forward pass is small enough to run in a browser tab
 * without a tensor library.
 */

export const LN_EPS = 1e-5;

export function zeros(n: number): Float32Array {
  return new Float32Array(n);
}

/** out(m x n) = a(m x k) @ b(k x n) */
export function matmul(
  a: Float32Array,
  m: number,
  k: number,
  b: Float32Array,
  n: number,
  out: Float32Array,
  accumulate = false,
): void {
  if (!accumulate) out.fill(0, 0, m * n);
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const oRow = i * n;
    for (let p = 0; p < k; p++) {
      const av = a[aRow + p];
      if (av === 0) continue;
      const bRow = p * n;
      for (let j = 0; j < n; j++) out[oRow + j] += av * b[bRow + j];
    }
  }
}

/** out(m x n) = a(k x m)^T @ b(k x n) */
export function matmulAT(
  a: Float32Array,
  k: number,
  m: number,
  b: Float32Array,
  n: number,
  out: Float32Array,
  accumulate = false,
): void {
  if (!accumulate) out.fill(0, 0, m * n);
  for (let p = 0; p < k; p++) {
    const aRow = p * m;
    const bRow = p * n;
    for (let i = 0; i < m; i++) {
      const av = a[aRow + i];
      if (av === 0) continue;
      const oRow = i * n;
      for (let j = 0; j < n; j++) out[oRow + j] += av * b[bRow + j];
    }
  }
}

/** out(m x n) = a(m x k) @ b(n x k)^T */
export function matmulBT(
  a: Float32Array,
  m: number,
  k: number,
  b: Float32Array,
  n: number,
  out: Float32Array,
  accumulate = false,
): void {
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const oRow = i * n;
    for (let j = 0; j < n; j++) {
      const bRow = j * k;
      let acc = 0;
      for (let p = 0; p < k; p++) acc += a[aRow + p] * b[bRow + p];
      if (accumulate) out[oRow + j] += acc;
      else out[oRow + j] = acc;
    }
  }
}

/**
 * out(m x m) = strictly-lower-triangular part of a(m x k) @ a(m x k)^T.
 * This is bdh.py's `(QR @ KR.mT).tril(diagonal=-1)`: token i attends only to
 * strictly earlier tokens, never to itself.
 */
export function scoresStrictLowerTri(
  a: Float32Array,
  m: number,
  k: number,
  out: Float32Array,
): void {
  out.fill(0, 0, m * m);
  for (let i = 1; i < m; i++) {
    const aRow = i * k;
    const oRow = i * m;
    for (let j = 0; j < i; j++) {
      const bRow = j * k;
      let acc = 0;
      for (let p = 0; p < k; p++) acc += a[aRow + p] * a[bRow + p];
      out[oRow + j] = acc;
    }
  }
}

/** out(n) = a(m x n)^T @ v(m) */
export function matvecAT(
  a: Float32Array,
  m: number,
  n: number,
  v: Float32Array,
  out: Float32Array,
): void {
  out.fill(0, 0, n);
  for (let i = 0; i < m; i++) {
    const av = v[i];
    if (av === 0) continue;
    const aRow = i * n;
    for (let j = 0; j < n; j++) out[j] += av * a[aRow + j];
  }
}

/** out(m) = a(m x n) @ v(n) */
export function matvec(
  a: Float32Array,
  m: number,
  n: number,
  v: Float32Array,
  out: Float32Array,
): void {
  for (let i = 0; i < m; i++) {
    const aRow = i * n;
    let acc = 0;
    for (let j = 0; j < n; j++) acc += a[aRow + j] * v[j];
    out[i] = acc;
  }
}

/** acc(m x n) += u(m) (outer) v(n) — the rank-1 synaptic update. */
export function addOuter(
  acc: Float32Array,
  u: Float32Array,
  m: number,
  v: Float32Array,
  n: number,
): void {
  for (let i = 0; i < m; i++) {
    const uv = u[i];
    if (uv === 0) continue; // sparse rows cost nothing: only firing neurons write state
    const row = i * n;
    for (let j = 0; j < n; j++) acc[row + j] += uv * v[j];
  }
}

/** LayerNorm with no affine parameters (matches nn.LayerNorm(elementwise_affine=False)). */
export function layerNorm(
  src: Float32Array,
  srcOff: number,
  d: number,
  dst: Float32Array,
  dstOff: number,
): void {
  let mean = 0;
  for (let i = 0; i < d; i++) mean += src[srcOff + i];
  mean /= d;
  let variance = 0;
  for (let i = 0; i < d; i++) {
    const c = src[srcOff + i] - mean;
    variance += c * c;
  }
  variance /= d;
  const inv = 1 / Math.sqrt(variance + LN_EPS);
  for (let i = 0; i < d; i++) dst[dstOff + i] = (src[srcOff + i] - mean) * inv;
}

/**
 * Gradient of a no-affine LayerNorm.
 * `normed` is the LN output, `dOut` the incoming gradient; writes d(input).
 */
export function layerNormBackward(
  normed: Float32Array,
  normedOff: number,
  invStd: number,
  dOut: Float32Array,
  dOutOff: number,
  d: number,
  dIn: Float32Array,
  dInOff: number,
): void {
  let meanDOut = 0;
  let meanDOutY = 0;
  for (let i = 0; i < d; i++) {
    const g = dOut[dOutOff + i];
    meanDOut += g;
    meanDOutY += g * normed[normedOff + i];
  }
  meanDOut /= d;
  meanDOutY /= d;
  for (let i = 0; i < d; i++) {
    dIn[dInOff + i] =
      invStd *
      (dOut[dOutOff + i] - meanDOut - normed[normedOff + i] * meanDOutY);
  }
}

/** 1 / sqrt(var + eps) for a slice, needed to replay LayerNorm gradients. */
export function invStdOf(src: Float32Array, off: number, d: number): number {
  let mean = 0;
  for (let i = 0; i < d; i++) mean += src[off + i];
  mean /= d;
  let variance = 0;
  for (let i = 0; i < d; i++) {
    const c = src[off + i] - mean;
    variance += c * c;
  }
  return 1 / Math.sqrt(variance / d + LN_EPS);
}

export function countPositive(v: Float32Array, off: number, len: number): number {
  let c = 0;
  for (let i = 0; i < len; i++) if (v[off + i] > 0) c++;
  return c;
}

/** Deterministic PRNG so every visitor sees the same "hatchling". */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
