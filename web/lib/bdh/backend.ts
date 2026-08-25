/**
 * One interface over the two implementations of the same model, so the lab does
 * not care which one it is driving.
 *
 * WebGPU is used whenever the browser has it and agrees with the CPU reference;
 * otherwise the CPU path runs. There is no switch to flip — a backend that
 * cannot be trusted is not used.
 */

import type { ToyConfig, ToyParams } from "./model";
import { ToySession, type StepTrace } from "./recurrent";
import { compareToCpu, type BackendComparison } from "./selfcheck";
import { requestGpuContext, WebGpuSession, WebGpuUnavailable } from "./webgpu/backend";

export type BackendKind = "webgpu" | "cpu";

export interface BdhBackend {
  readonly kind: BackendKind;
  /** what the UI shows: "WebGPU" or "CPU" */
  readonly label: string;
  /** adapter description for WebGPU, or why WebGPU was not used */
  readonly detail: string | null;
  readonly position: number;
  reset(): void;
  step(token: number, wantTraces?: boolean): Promise<StepTrace>;
  dispose(): void;
}

class CpuBackend implements BdhBackend {
  readonly kind = "cpu" as const;
  readonly label = "CPU";
  detail: string | null;
  private session: ToySession;
  private readonly params: ToyParams;
  private readonly config: ToyConfig;

  constructor(params: ToyParams, config: ToyConfig, detail: string | null) {
    this.params = params;
    this.config = config;
    this.detail = detail;
    this.session = new ToySession(params, config);
  }

  get position(): number {
    return this.session.position;
  }

  reset(): void {
    this.session = new ToySession(this.params, this.config);
  }

  async step(token: number, wantTraces = true): Promise<StepTrace> {
    return this.session.feed(token, wantTraces);
  }

  dispose(): void {}
}

class GpuBackend implements BdhBackend {
  readonly kind = "webgpu" as const;
  readonly label = "WebGPU";
  readonly detail: string | null;
  private readonly session: WebGpuSession;

  constructor(session: WebGpuSession) {
    this.session = session;
    this.detail = session.adapterLabel;
  }

  get position(): number {
    return this.session.position;
  }

  reset(): void {
    this.session.reset();
  }

  step(token: number, wantTraces = true): Promise<StepTrace> {
    return this.session.step(token, wantTraces);
  }

  dispose(): void {
    this.session.dispose();
  }
}

export interface BackendSelection {
  backend: BdhBackend;
  /** null when WebGPU was not usable */
  comparison: BackendComparison | null;
  /** why the CPU path was chosen, when it was */
  fallbackReason: string | null;
}

/**
 * Reject a GPU whose logits disagree with the CPU reference by more than this,
 * relative to the logit scale. Set well above the float32 noise floor: it is
 * here to catch a broken driver, not to police rounding.
 */
export const GPU_TRUST_THRESHOLD = 1e-2;

/** Bytes fed to both backends when checking that they agree. */
export const SELFCHECK_PROMPT = "To be or not to be, that is the question.\n";

export async function createBackend(
  params: ToyParams,
  config: ToyConfig,
): Promise<BackendSelection> {
  let session: WebGpuSession | null = null;
  try {
    const context = await requestGpuContext();
    session = await WebGpuSession.create(params, config, context);

    // Prove the GPU agrees with the verified CPU path before trusting it.
    const comparison = await compareToCpu(session, params, config, SELFCHECK_PROMPT);
    if (!(comparison.relativeToScale < GPU_TRUST_THRESHOLD)) {
      session.dispose();
      return {
        backend: new CpuBackend(
          params,
          config,
          `WebGPU output disagreed with the CPU reference (relative deviation ${comparison.relativeToScale.toExponential(2)})`,
        ),
        comparison,
        fallbackReason: "WebGPU disagreed with the CPU reference",
      };
    }

    session.reset();
    return {
      backend: new GpuBackend(session),
      comparison,
      fallbackReason: null,
    };
  } catch (cause) {
    session?.dispose();
    const reason =
      cause instanceof WebGpuUnavailable
        ? cause.message
        : `WebGPU setup failed: ${cause instanceof Error ? cause.message : String(cause)}`;
    return {
      backend: new CpuBackend(params, config, reason),
      comparison: null,
      fallbackReason: reason,
    };
  }
}

/** Used when the GPU throws mid-run: swap in the CPU path and keep going. */
export function createCpuBackend(
  params: ToyParams,
  config: ToyConfig,
  reason: string,
): BdhBackend {
  return new CpuBackend(params, config, reason);
}
