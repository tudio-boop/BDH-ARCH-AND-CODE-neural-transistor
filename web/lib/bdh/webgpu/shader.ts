/**
 * WGSL for one BDH-GPU token step.
 *
 * The whole step — all layers, then the readout — runs in a *single* workgroup
 * of `WG` threads, with `workgroupBarrier()` between stages. That is possible
 * because the toy is small: one thread per neuron (n = 256) covers every
 * n-sized stage, and the d-sized stages (d = 32) are handled by the first d
 * threads. Keeping it to one dispatch per token matters much more than raw
 * arithmetic throughput here, since a token is only ~170k multiply-adds and
 * dispatch overhead would otherwise dominate completely.
 *
 * The reduction order is deliberately identical to the CPU implementation in
 * ../recurrent.ts, so the only systematic difference is that the GPU
 * accumulates in f32 where JavaScript numbers accumulate in f64.
 *
 * Positions arrive as a precomputed cos/sin table rather than being derived in
 * the shader: `pos * freq` loses too much precision in f32 for the resulting
 * phase to match the CPU path, and the table is only 2n floats per token.
 */

import type { ToyConfig } from "../model";

export const WORKGROUP_SIZE = 256;

export interface ShaderLayout {
  /** offsets into the packed parameter buffer, in f32 elements */
  params: {
    embed: number;
    Dx: number;
    Dy: number;
    E: number;
    head: number;
    total: number;
  };
  /** offsets into the output buffer, in f32 elements */
  out: {
    logits: number;
    traces: number;
    traceStride: number;
    total: number;
  };
  rhoFloats: number;
  ropeFloats: number;
}

export function shaderLayout(c: ToyConfig): ShaderLayout {
  const embed = 0;
  const Dx = embed + c.vocab * c.d;
  const Dy = Dx + c.d * c.n;
  const E = Dy + c.d * c.n;
  const head = E + c.n * c.d;
  const total = head + c.d * c.vocab;

  const traceStride = 3 * c.n;
  return {
    params: { embed, Dx, Dy, E, head, total },
    out: {
      logits: 0,
      traces: c.vocab,
      traceStride,
      total: c.vocab + c.layers * traceStride,
    },
    rhoFloats: c.layers * c.n * c.d,
    ropeFloats: 2 * c.n,
  };
}

export function buildShader(c: ToyConfig): string {
  const l = shaderLayout(c);
  return `// generated for BDH-GPU(n=${c.n}, d=${c.d}), ${c.layers} layers, vocab ${c.vocab}

const WG: u32 = ${WORKGROUP_SIZE}u;
const D: u32 = ${c.d}u;
const N: u32 = ${c.n}u;
const LAYERS: u32 = ${c.layers}u;
const VOCAB: u32 = ${c.vocab}u;
const LN_EPS: f32 = 1e-5;

const OFF_EMBED: u32 = ${l.params.embed}u;
const OFF_DX: u32 = ${l.params.Dx}u;
const OFF_DY: u32 = ${l.params.Dy}u;
const OFF_E: u32 = ${l.params.E}u;
const OFF_HEAD: u32 = ${l.params.head}u;

const OUT_LOGITS: u32 = ${l.out.logits}u;
const OUT_TRACES: u32 = ${l.out.traces}u;
const TRACE_STRIDE: u32 = ${l.out.traceStride}u;

struct Control {
  token: u32,
  position: u32,
  writeTraces: u32,
  reserved: u32,
};

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> rho: array<f32>;
// [cos(phase_0..phase_N-1), sin(phase_0..phase_N-1)] for the current position
@group(0) @binding(2) var<storage, read> rope: array<f32>;
@group(0) @binding(3) var<uniform> control: Control;
@group(0) @binding(4) var<storage, read_write> results: array<f32>;

var<workgroup> sx: array<f32, D>;
var<workgroup> sxIn: array<f32, D>;
var<workgroup> sxs: array<f32, N>;
var<workgroup> sqr: array<f32, N>;
var<workgroup> sa: array<f32, D>;
var<workgroup> san: array<f32, D>;
var<workgroup> sgate: array<f32, N>;
var<workgroup> sy: array<f32, N>;
var<workgroup> sym: array<f32, D>;
var<workgroup> syn: array<f32, D>;
var<workgroup> ssum: array<f32, D>;

// Mean and 1/sqrt(var + eps) of a d-vector, matching nn.LayerNorm with no
// affine parameters. Taken by value: WGSL restricts workgroup pointers as
// function parameters, and copying ${c.d} floats is cheaper than the
// alternatives. Every thread computes the same statistics redundantly, which
// keeps the barriers in uniform control flow.
fn ln_stats(v: array<f32, D>) -> vec2<f32> {
  var mean = 0.0;
  for (var k = 0u; k < D; k = k + 1u) {
    mean = mean + v[k];
  }
  mean = mean / f32(D);
  var variance = 0.0;
  for (var k = 0u; k < D; k = k + 1u) {
    let centered = v[k] - mean;
    variance = variance + centered * centered;
  }
  variance = variance / f32(D);
  return vec2<f32>(mean, 1.0 / sqrt(variance + LN_EPS));
}

@compute @workgroup_size(WG)
fn step(@builtin(local_invocation_id) local_id: vec3<u32>) {
  let lid = local_id.x;

  // x = LN(embed[token])
  for (var j = lid; j < D; j = j + WG) {
    ssum[j] = weights[OFF_EMBED + control.token * D + j];
  }
  workgroupBarrier();
  {
    let st = ln_stats(ssum);
    for (var j = lid; j < D; j = j + WG) {
      sx[j] = (ssum[j] - st.x) * st.y;
    }
  }
  workgroupBarrier();

  for (var layer = 0u; layer < LAYERS; layer = layer + 1u) {
    let rhoBase = layer * N * D;

    // the layer input doubles as the attention value, so keep a copy
    for (var j = lid; j < D; j = j + WG) {
      sxIn[j] = sx[j];
    }
    workgroupBarrier();

    // x_sparse = ReLU(x @ Dx): lift into neuron space, keep positives
    for (var i = lid; i < N; i = i + WG) {
      var acc = 0.0;
      for (var k = 0u; k < D; k = k + 1u) {
        acc = acc + sx[k] * weights[OFF_DX + k * N + i];
      }
      sxs[i] = max(acc, 0.0);
    }
    workgroupBarrier();

    // Q = K = RoPE(x_sparse): pairs of neurons share a rotation
    for (var i = lid; i < N; i = i + WG) {
      let c = rope[i];
      let s = rope[N + i];
      if ((i & 1u) == 0u) {
        sqr[i] = sxs[i] * c - sxs[i + 1u] * s;
      } else {
        sqr[i] = sxs[i] * c + sxs[i - 1u] * s;
      }
    }
    workgroupBarrier();

    // read working memory: a = rho^T q
    for (var j = lid; j < D; j = j + WG) {
      var acc = 0.0;
      for (var i = 0u; i < N; i = i + 1u) {
        acc = acc + sqr[i] * rho[rhoBase + i * D + j];
      }
      sa[j] = acc;
    }
    workgroupBarrier();
    {
      let st = ln_stats(sa);
      for (var j = lid; j < D; j = j + WG) {
        san[j] = (sa[j] - st.x) * st.y;
      }
    }
    workgroupBarrier();

    // gate = ReLU(LN(a) @ Dy), then y = gate * x_sparse
    for (var i = lid; i < N; i = i + WG) {
      var acc = 0.0;
      for (var k = 0u; k < D; k = k + 1u) {
        acc = acc + san[k] * weights[OFF_DY + k * N + i];
      }
      let gated = max(acc, 0.0);
      sgate[i] = gated;
      sy[i] = gated * sxs[i];
    }
    workgroupBarrier();

    // back down through E
    for (var j = lid; j < D; j = j + WG) {
      var acc = 0.0;
      for (var i = 0u; i < N; i = i + 1u) {
        acc = acc + sy[i] * weights[OFF_E + i * D + j];
      }
      sym[j] = acc;
    }
    workgroupBarrier();
    {
      let st = ln_stats(sym);
      for (var j = lid; j < D; j = j + WG) {
        syn[j] = (sym[j] - st.x) * st.y;
      }
    }
    workgroupBarrier();
    for (var j = lid; j < D; j = j + WG) {
      ssum[j] = sxIn[j] + syn[j];
    }
    workgroupBarrier();
    {
      let st = ln_stats(ssum);
      for (var j = lid; j < D; j = j + WG) {
        sx[j] = (ssum[j] - st.x) * st.y;
      }
    }
    workgroupBarrier();

    // write working memory: rho += q (outer) x_in.
    // A silent neuron leaves its whole row untouched.
    for (var i = lid; i < N; i = i + WG) {
      let q = sqr[i];
      if (q != 0.0) {
        let base = rhoBase + i * D;
        for (var j = 0u; j < D; j = j + 1u) {
          rho[base + j] = rho[base + j] + q * sxIn[j];
        }
      }
    }

    if (control.writeTraces == 1u) {
      let traceBase = OUT_TRACES + layer * TRACE_STRIDE;
      for (var i = lid; i < N; i = i + WG) {
        results[traceBase + i] = sxs[i];
        results[traceBase + N + i] = sgate[i];
        results[traceBase + 2u * N + i] = sy[i];
      }
    }
    workgroupBarrier();
  }

  // logits = x @ lm_head
  for (var v = lid; v < VOCAB; v = v + WG) {
    var acc = 0.0;
    for (var k = 0u; k < D; k = k + 1u) {
      acc = acc + sx[k] * weights[OFF_HEAD + k * VOCAB + v];
    }
    results[OUT_LOGITS + v] = acc;
  }
}
`;
}
