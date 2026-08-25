/**
 * WebGPU implementation of the BDH-GPU token step.
 *
 * One dispatch per token, one workgroup, everything resident on the device:
 * the weights and the rho state are uploaded once, and each token only writes a
 * 16-byte control block plus the position's rotation table. What comes back is
 * the logits and the per-layer activations the lab draws.
 *
 * Numerics are meant to match ../recurrent.ts to float32 rounding. See
 * ../selfcheck.ts, which measures the difference rather than assuming it.
 */

import { countPositive } from "../linalg";
import { ropeFreqs, type ToyConfig, type ToyParams } from "../model";
import type { LayerTrace, StepTrace } from "../recurrent";
import { buildShader, shaderLayout, WORKGROUP_SIZE, type ShaderLayout } from "./shader";

export class WebGpuUnavailable extends Error {}

export interface GpuContext {
  device: GPUDevice;
  /** best available human-readable description of the adapter */
  adapterLabel: string | null;
}

/** Feature-detect and acquire a device, with a specific reason on failure. */
export async function requestGpuContext(): Promise<GpuContext> {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new WebGpuUnavailable("this browser does not expose navigator.gpu");
  }

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (cause) {
    throw new WebGpuUnavailable(`requestAdapter failed: ${describe(cause)}`);
  }
  if (!adapter) {
    throw new WebGpuUnavailable("no WebGPU adapter is available");
  }

  if (adapter.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE) {
    throw new WebGpuUnavailable(
      `adapter allows only ${adapter.limits.maxComputeInvocationsPerWorkgroup} invocations per workgroup, need ${WORKGROUP_SIZE}`,
    );
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice();
  } catch (cause) {
    throw new WebGpuUnavailable(`requestDevice failed: ${describe(cause)}`);
  }

  const info = adapter.info;
  const parts = info
    ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean)
    : [];
  return { device, adapterLabel: parts.length > 0 ? parts.join(" ") : null };
}

function describe(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

export class WebGpuSession {
  readonly config: ToyConfig;
  readonly adapterLabel: string | null;
  private readonly device: GPUDevice;
  private readonly layout: ShaderLayout;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly buffers: {
    weights: GPUBuffer;
    rho: GPUBuffer;
    rope: GPUBuffer;
    control: GPUBuffer;
    results: GPUBuffer;
    staging: GPUBuffer;
  };
  private readonly freqs: Float32Array;
  private readonly ropeHost: Float32Array;
  private readonly controlHost: Uint32Array;
  private readonly zeroRho: Float32Array;
  private deviceError: string | null = null;
  private disposed = false;

  position = 0;

  private constructor(
    device: GPUDevice,
    adapterLabel: string | null,
    config: ToyConfig,
    layout: ShaderLayout,
    pipeline: GPUComputePipeline,
    buffers: WebGpuSession["buffers"],
  ) {
    this.device = device;
    this.adapterLabel = adapterLabel;
    this.config = config;
    this.layout = layout;
    this.pipeline = pipeline;
    this.buffers = buffers;
    this.freqs = ropeFreqs(config.n, config.ropeTheta);
    this.ropeHost = new Float32Array(layout.ropeFloats);
    this.controlHost = new Uint32Array(4);
    this.zeroRho = new Float32Array(layout.rhoFloats);
    this.bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.weights } },
        { binding: 1, resource: { buffer: buffers.rho } },
        { binding: 2, resource: { buffer: buffers.rope } },
        { binding: 3, resource: { buffer: buffers.control } },
        { binding: 4, resource: { buffer: buffers.results } },
      ],
    });

    // Not every WebGPU host implements the diagnostic surface (Deno, used by
    // scripts/check-webgpu.ts, does not expose these), so probe before using.
    if (typeof device.addEventListener === "function") {
      device.addEventListener("uncapturederror", (event) => {
        const error = (event as GPUUncapturedErrorEvent).error;
        this.deviceError ??= error.message;
      });
    }
    if (device.lost && typeof device.lost.then === "function") {
      void device.lost.then((info) => {
        this.deviceError ??= `device lost: ${info.message || info.reason}`;
      });
    }
  }

  static async create(
    params: ToyParams,
    config: ToyConfig,
    context?: GpuContext,
  ): Promise<WebGpuSession> {
    const { device, adapterLabel } = context ?? (await requestGpuContext());
    const layout = shaderLayout(config);

    const usesErrorScopes = typeof device.pushErrorScope === "function";
    if (usesErrorScopes) device.pushErrorScope("validation");

    const module = device.createShaderModule({
      code: buildShader(config),
      label: "bdh-step",
    });

    if (typeof module.getCompilationInfo === "function") {
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        if (usesErrorScopes) await device.popErrorScope();
        throw new WebGpuUnavailable(
          `shader failed to compile: ${errors
            .map((e) => `line ${e.lineNum}: ${e.message}`)
            .join("; ")}`,
        );
      }
    }

    const descriptor: GPUComputePipelineDescriptor = {
      layout: "auto",
      compute: { module, entryPoint: "step" },
    };
    const pipeline =
      typeof device.createComputePipelineAsync === "function"
        ? await device.createComputePipelineAsync(descriptor)
        : device.createComputePipeline(descriptor);

    if (usesErrorScopes) {
      const validationError = await device.popErrorScope();
      if (validationError) {
        throw new WebGpuUnavailable(`pipeline invalid: ${validationError.message}`);
      }
    }

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const buffers = {
      weights: device.createBuffer({
        label: "bdh-weights",
        size: layout.params.total * 4,
        usage: storage,
      }),
      rho: device.createBuffer({
        label: "bdh-rho",
        size: layout.rhoFloats * 4,
        usage: storage,
      }),
      rope: device.createBuffer({
        label: "bdh-rope",
        size: layout.ropeFloats * 4,
        usage: storage,
      }),
      control: device.createBuffer({
        label: "bdh-control",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      results: device.createBuffer({
        label: "bdh-results",
        size: layout.out.total * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      staging: device.createBuffer({
        label: "bdh-staging",
        size: layout.out.total * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };

    const session = new WebGpuSession(
      device,
      adapterLabel,
      config,
      layout,
      pipeline,
      buffers,
    );
    session.uploadWeights(params);
    session.reset();
    return session;
  }

  private uploadWeights(params: ToyParams): void {
    const packed = new Float32Array(this.layout.params.total);
    packed.set(params.embed, this.layout.params.embed);
    packed.set(params.Dx, this.layout.params.Dx);
    packed.set(params.Dy, this.layout.params.Dy);
    packed.set(params.E, this.layout.params.E);
    packed.set(params.head, this.layout.params.head);
    this.device.queue.writeBuffer(this.buffers.weights, 0, packed);
  }

  /** Clear working memory and go back to position 0. */
  reset(): void {
    this.device.queue.writeBuffer(this.buffers.rho, 0, this.zeroRho);
    this.position = 0;
  }

  /**
   * Rotation table for one position, computed on the CPU exactly as
   * model.ts:ropeApply does, because f32 cannot hold `pos * freq` precisely
   * enough for the phases to agree.
   */
  private fillRope(position: number): void {
    const n = this.config.n;
    for (let j = 0; j < n; j += 2) {
      let phase = (position * this.freqs[j]) % 1;
      if (phase < 0) phase += 1;
      phase *= 2 * Math.PI;
      const c = Math.cos(phase);
      const s = Math.sin(phase);
      this.ropeHost[j] = c;
      this.ropeHost[j + 1] = c;
      this.ropeHost[n + j] = s;
      this.ropeHost[n + j + 1] = s;
    }
  }

  async step(token: number, wantTraces = true): Promise<StepTrace> {
    if (this.disposed) throw new Error("session disposed");
    const position = this.position;

    this.fillRope(position);
    this.device.queue.writeBuffer(this.buffers.rope, 0, this.ropeHost);
    this.controlHost[0] = token;
    this.controlHost[1] = position;
    this.controlHost[2] = wantTraces ? 1 : 0;
    this.controlHost[3] = 0;
    this.device.queue.writeBuffer(this.buffers.control, 0, this.controlHost);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(
      this.buffers.results,
      0,
      this.buffers.staging,
      0,
      this.layout.out.total * 4,
    );
    this.device.queue.submit([encoder.finish()]);

    await this.buffers.staging.mapAsync(GPUMapMode.READ);
    const copy = this.buffers.staging.getMappedRange().slice(0);
    this.buffers.staging.unmap();

    if (this.deviceError) {
      throw new Error(this.deviceError);
    }

    this.position = position + 1;
    return this.readTrace(new Float32Array(copy), token, position, wantTraces);
  }

  private readTrace(
    values: Float32Array,
    token: number,
    position: number,
    wantTraces: boolean,
  ): StepTrace {
    const { n, layers } = this.config;
    const logits = values.slice(
      this.layout.out.logits,
      this.layout.out.logits + this.config.vocab,
    );
    const traces: LayerTrace[] = [];
    if (wantTraces) {
      for (let l = 0; l < layers; l++) {
        const base = this.layout.out.traces + l * this.layout.out.traceStride;
        const x = values.slice(base, base + n);
        const gate = values.slice(base + n, base + 2 * n);
        const y = values.slice(base + 2 * n, base + 3 * n);
        traces.push({
          x,
          gate,
          y,
          xPositive: countPositive(x, 0, n),
          gatePositive: countPositive(gate, 0, n),
          yPositive: countPositive(y, 0, n),
        });
      }
    }
    return { position, token, layers: traces, logits };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
    this.device.destroy();
  }
}
