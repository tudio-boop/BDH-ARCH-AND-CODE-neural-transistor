/**
 * Packing for the trained toy weights.
 *
 * The demo ships real trained weights, so they have to be small: int8 with one
 * scale per row, base64'd into JSON. scripts/train-toy.ts measures the
 * validation loss before and after packing and records both in the file, so the
 * cost of the compression is visible rather than assumed.
 */

import {
  paramShapes,
  PARAM_KEYS,
  type ParamKey,
  type ToyConfig,
  type ToyParams,
} from "./model";

export const WEIGHTS_FORMAT = "bdh-toy-int8-rowscale-v1";

export interface PackedTensor {
  shape: [number, number];
  /** one scale per row of the tensor */
  scales: number[];
  /** base64 of the int8 values, row-major */
  data: string;
}

export interface ToyWeightsFile {
  format: string;
  note: string;
  config: ToyConfig;
  shapes: Record<ParamKey, [number, number]>;
  training: {
    parameters: number;
    steps: number;
    batch: number;
    block: number;
    lr: number;
    optimizer: string;
    schedule: string;
    weightDecay: number;
    firstLoss: number;
    trainLoss: number;
    valLoss: number;
    valLossInt8: number;
    history: { step: number; loss: number }[];
    corpusBytes: number;
    positiveActivations: { x: number; gate: number; y: number };
    seconds: number;
  };
  tensors: Record<ParamKey, PackedTensor>;
}

function toBase64(bytes: Int8Array): string {
  const view = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < view.length; i += chunk) {
    binary += String.fromCharCode(...view.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function fromBase64(text: string): Int8Array {
  const binary = atob(text);
  const out = new Int8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    const c = binary.charCodeAt(i);
    out[i] = c > 127 ? c - 256 : c;
  }
  return out;
}

export function quantizeParams(
  params: ToyParams,
  config: ToyConfig,
): { tensors: Record<ParamKey, PackedTensor>; dequantized: ToyParams } {
  const shapes = paramShapes(config);
  const tensors = {} as Record<ParamKey, PackedTensor>;
  const dequantized = {} as ToyParams;

  for (const key of PARAM_KEYS as readonly ParamKey[]) {
    const [rows, cols] = shapes[key];
    const src = params[key];
    const q = new Int8Array(rows * cols);
    const scales: number[] = new Array(rows);
    const back = new Float32Array(rows * cols);

    for (let r = 0; r < rows; r++) {
      const off = r * cols;
      let max = 0;
      for (let c = 0; c < cols; c++) {
        const a = Math.abs(src[off + c]);
        if (a > max) max = a;
      }
      const scale = max > 0 ? max / 127 : 1;
      scales[r] = scale;
      for (let c = 0; c < cols; c++) {
        const v = Math.round(src[off + c] / scale);
        q[off + c] = Math.max(-127, Math.min(127, v));
        back[off + c] = q[off + c] * scale;
      }
    }

    tensors[key] = { shape: [rows, cols], scales, data: toBase64(q) };
    dequantized[key] = back;
  }

  return { tensors, dequantized };
}

export function dequantizeParams(file: ToyWeightsFile): ToyParams {
  const out = {} as ToyParams;
  for (const key of PARAM_KEYS as readonly ParamKey[]) {
    const tensor = file.tensors[key];
    const [rows, cols] = tensor.shape;
    const q = fromBase64(tensor.data);
    if (q.length !== rows * cols) {
      throw new Error(
        `weights for ${key} are ${q.length} values, expected ${rows * cols}`,
      );
    }
    const values = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const off = r * cols;
      const scale = tensor.scales[r];
      for (let c = 0; c < cols; c++) values[off + c] = q[off + c] * scale;
    }
    out[key] = values;
  }
  return out;
}

export const TOY_WEIGHTS_URL = "/bdh-toy-weights.json";

export async function loadToyWeights(
  signal?: AbortSignal,
): Promise<{ file: ToyWeightsFile; params: ToyParams }> {
  const res = await fetch(TOY_WEIGHTS_URL, { signal });
  if (!res.ok) throw new Error(`could not load toy weights (${res.status})`);
  const file = (await res.json()) as ToyWeightsFile;
  if (file.format !== WEIGHTS_FORMAT) {
    throw new Error(`unexpected weights format ${file.format}`);
  }
  return { file, params: dequantizeParams(file) };
}
