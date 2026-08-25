/**
 * WebGPU vs CPU numerical check, outside the browser.
 *
 * This runs under **Deno**, which ships a WebGPU implementation, so the GPU
 * path can be checked from a terminal without driving a browser and without
 * adding a native dependency to this project. Deno is not required to build,
 * deploy or use the site; the same comparison runs in the browser on /verify
 * and on every visit to /lab.
 *
 *   deno run --allow-read --unstable-webgpu --unstable-sloppy-imports \
 *     --no-check scripts/check-webgpu.ts
 *
 * On Linux without a GPU, point Deno at a software Vulkan device, for example
 * Chrome's bundled SwiftShader:
 *
 *   VK_ICD_FILENAMES=/opt/google/chrome/vk_swiftshader_icd.json deno run ...
 */

import { TOY_CONFIG } from "../lib/bdh/model";
import { compareToCpu } from "../lib/bdh/selfcheck";
import { dequantizeParams, type ToyWeightsFile } from "../lib/bdh/weights";
import { requestGpuContext, WebGpuSession } from "../lib/bdh/webgpu/backend";

const WEIGHTS_PATH = new URL("../public/bdh-toy-weights.json", import.meta.url);

// A long-ish, varied prompt: more positions means more chances for f32 to drift.
const PROMPT =
  "To be or not to be, that is the question.\n" +
  "Whether 'tis nobler in the mind to suffer\n" +
  "The slings and arrows of outrageous fortune.\n";

const TOLERANCE = 1e-3;

function requireDeno(): void {
  if (typeof (globalThis as { Deno?: unknown }).Deno === "undefined") {
    console.error(
      "This check needs a WebGPU host. Run it with Deno:\n" +
        "  deno run --allow-read --unstable-webgpu --unstable-sloppy-imports --no-check scripts/check-webgpu.ts\n" +
        "Or open /verify in a browser that has WebGPU.",
    );
    throw new Error("no WebGPU host");
  }
}

async function main(): Promise<void> {
  requireDeno();

  const file = JSON.parse(
    await (globalThis as unknown as {
      Deno: { readTextFile(path: URL): Promise<string> };
    }).Deno.readTextFile(WEIGHTS_PATH),
  ) as ToyWeightsFile;
  const params = dequantizeParams(file);
  const config = TOY_CONFIG;

  const context = await requestGpuContext();
  console.log(
    `adapter: ${context.adapterLabel ?? "(no description reported)"}`,
  );
  console.log(
    `model:   BDH-GPU(n=${config.n}, d=${config.d}), ${config.layers} layers, ` +
      `${file.training.parameters.toLocaleString("en-US")} parameters`,
  );

  const session = await WebGpuSession.create(params, config, context);
  const result = await compareToCpu(session, params, config, PROMPT);
  session.dispose();

  console.log(`\nteacher-forced over ${result.tokens} bytes`);
  console.log(`  largest |logit| (CPU)        ${result.logitScale.toExponential(4)}`);
  console.log(`  max |GPU - CPU| logit        ${result.maxAbsolute.toExponential(4)}`);
  console.log(`  relative to the logit scale  ${result.relativeToScale.toExponential(4)}`);
  console.log(`  mean |GPU - CPU| logit       ${result.meanAbsolute.toExponential(4)}`);
  console.log(`  max |GPU - CPU| in y         ${result.maxActivationDifference.toExponential(4)}`);
  console.log(`  positions where top-1 byte differed  ${result.argmaxDisagreements}/${result.tokens}`);

  if (!(result.relativeToScale < TOLERANCE)) {
    console.error(
      `\nFAIL: relative deviation ${result.relativeToScale.toExponential(3)} exceeds ${TOLERANCE}`,
    );
    throw new Error("backends disagree");
  }
  console.log("\nPASS: WebGPU matches the CPU reference");
}

await main();
