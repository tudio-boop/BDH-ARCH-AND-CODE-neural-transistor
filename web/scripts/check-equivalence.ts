/**
 * The demo's claim is that carrying a fixed-size state rho gives the same
 * answer as the token-parallel triangular attention in bdh.py. This asserts it.
 *
 *   npm run toy:check
 */

import { mulberry32 } from "../lib/bdh/linalg";
import { initParams, ropeFreqs, TOY_CONFIG } from "../lib/bdh/model";
import { allocCache, forward } from "../lib/bdh/parallel";
import { createRhoState, rhoFloats, kvCacheFloats, ToySession } from "../lib/bdh/recurrent";

const config = TOY_CONFIG;
const T = 48;
const params = initParams(config, 4242);
const freqs = ropeFreqs(config.n, config.ropeTheta);

const rand = mulberry32(11);
const tokens = new Uint8Array(T);
for (let t = 0; t < T; t++) tokens[t] = Math.floor(rand() * config.vocab);

// Token-parallel: the whole T x T score matrix at once.
const cache = allocCache(config, T);
forward(params, config, freqs, tokens, cache);

// Recurrent: one token at a time, carrying rho.
const session = new ToySession(params, config);
const stepLogits: Float32Array[] = [];
for (let t = 0; t < T; t++) {
  stepLogits.push(session.feed(tokens[t], true).logits.slice());
}

let maxAbs = 0;
for (let t = 0; t < T; t++) {
  for (let v = 0; v < config.vocab; v++) {
    const abs = Math.abs(cache.logits[t * config.vocab + v] - stepLogits[t][v]);
    if (abs > maxAbs) maxAbs = abs;
  }
}

const scale = Math.max(
  ...Array.from(cache.logits.subarray(0, T * config.vocab), Math.abs),
);
const relative = maxAbs / scale;

console.log(`BDH-GPU(n=${config.n}, d=${config.d}), L=${config.layers}, T=${T}`);
console.log(`largest logit magnitude    ${scale.toExponential(3)}`);
console.log(`max |parallel - recurrent| ${maxAbs.toExponential(3)}`);
console.log(`relative to logit scale    ${relative.toExponential(3)}`);
console.log(
  `rho state: ${rhoFloats(config).toLocaleString()} floats, constant in T; ` +
    `a KV cache at T=${T} would be ${kvCacheFloats(config, T).toLocaleString()} and still growing`,
);

const occupancy = new ToySession(params, config);
for (let t = 0; t < T; t++) occupancy.feed(tokens[t]);
console.log(
  `rho entries written after ${T} tokens: ${occupancy
    .rhoOccupancy()
    .map((o) => `${(o * 100).toFixed(1)}%`)
    .join(", ")} per layer`,
);

// The two forms sum the same products in a different order, so float32 rounding
// is the only expected difference.
if (!(relative < 1e-4)) {
  console.error(`FAIL: recurrent and parallel forms disagree (${relative})`);
  process.exit(1);
}
console.log("PASS");

void createRhoState;
