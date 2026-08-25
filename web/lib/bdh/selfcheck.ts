/**
 * Does the WebGPU path compute the same thing as the CPU path?
 *
 * Both backends are fed the same bytes — teacher-forced, not two independent
 * generations — and the next-byte logits are compared at every position. That
 * distinction matters: sampling amplifies a 1e-6 logit difference into a
 * completely different sentence, so comparing generated text would measure the
 * sampler, not the arithmetic.
 *
 * The CPU path is the reference because it is the one checked against the
 * PyTorch model in bdh.py (see scripts/verify_against_bdh_py.py).
 */

import type { ToyConfig, ToyParams } from "./model";
import { encodeBytes, ToySession } from "./recurrent";
import type { WebGpuSession } from "./webgpu/backend";

export interface BackendComparison {
  tokens: number;
  /** largest absolute logit difference over all positions */
  maxAbsolute: number;
  /** that difference divided by the largest logit magnitude seen */
  relativeToScale: number;
  /** mean absolute difference, for context on the worst case */
  meanAbsolute: number;
  /** largest |logit| produced by the CPU reference */
  logitScale: number;
  /** worst per-layer difference in the y vectors the lab draws */
  maxActivationDifference: number;
  /** how often the two backends' top-1 next byte differed */
  argmaxDisagreements: number;
}

export async function compareToCpu(
  gpu: WebGpuSession,
  params: ToyParams,
  config: ToyConfig,
  prompt: string,
): Promise<BackendComparison> {
  const bytes = encodeBytes(prompt);
  const cpu = new ToySession(params, config);
  gpu.reset();

  let maxAbsolute = 0;
  let meanSum = 0;
  let count = 0;
  let logitScale = 0;
  let maxActivationDifference = 0;
  let argmaxDisagreements = 0;

  for (const token of bytes) {
    const cpuStep = cpu.feed(token, true);
    const gpuStep = await gpu.step(token, true);

    for (let v = 0; v < config.vocab; v++) {
      const reference = cpuStep.logits[v];
      const difference = Math.abs(reference - gpuStep.logits[v]);
      if (difference > maxAbsolute) maxAbsolute = difference;
      meanSum += difference;
      count += 1;
      const magnitude = Math.abs(reference);
      if (magnitude > logitScale) logitScale = magnitude;
    }

    if (argmax(cpuStep.logits) !== argmax(gpuStep.logits)) argmaxDisagreements += 1;

    for (let l = 0; l < cpuStep.layers.length; l++) {
      const a = cpuStep.layers[l].y;
      const b = gpuStep.layers[l].y;
      for (let i = 0; i < a.length; i++) {
        const difference = Math.abs(a[i] - b[i]);
        if (difference > maxActivationDifference) maxActivationDifference = difference;
      }
    }
  }

  return {
    tokens: bytes.length,
    maxAbsolute,
    relativeToScale: logitScale > 0 ? maxAbsolute / logitScale : maxAbsolute,
    meanAbsolute: count > 0 ? meanSum / count : 0,
    logitScale,
    maxActivationDifference,
    argmaxDisagreements,
  };
}

function argmax(values: Float32Array): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
