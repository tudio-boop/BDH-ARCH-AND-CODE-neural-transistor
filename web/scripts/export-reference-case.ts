/**
 * Dumps random weights, an input sequence, and this repo's TypeScript logits to
 * scripts/.cache/reference-case.json, so verify_against_bdh_py.py can feed the
 * exact same weights through the PyTorch BDH in ../bdh.py and compare.
 *
 *   npm run toy:export-ref && python3 scripts/verify_against_bdh_py.py
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { mulberry32 } from "../lib/bdh/linalg";
import { initParams, ropeFreqs, TOY_CONFIG, type ToyConfig } from "../lib/bdh/model";
import { allocCache, forward } from "../lib/bdh/parallel";
import { ToySession } from "../lib/bdh/recurrent";

const here = dirname(fileURLToPath(import.meta.url));

interface Case {
  name: string;
  config: ToyConfig;
  T: number;
  seed: number;
}

const cases: Case[] = [
  {
    name: "small",
    config: { d: 16, n: 64, layers: 3, vocab: 32, ropeTheta: 65536 },
    T: 24,
    seed: 5,
  },
  { name: "toy", config: TOY_CONFIG, T: 40, seed: 6 },
];

const payload = cases.map(({ name, config, T, seed }) => {
  const params = initParams(config, seed);
  const freqs = ropeFreqs(config.n, config.ropeTheta);
  const rand = mulberry32(seed * 31 + 1);
  const tokens = new Uint8Array(T);
  for (let t = 0; t < T; t++) tokens[t] = Math.floor(rand() * config.vocab);

  const cache = allocCache(config, T);
  forward(params, config, freqs, tokens, cache);

  const session = new ToySession(params, config);
  const recurrent: number[] = [];
  for (let t = 0; t < T; t++) {
    recurrent.push(...session.feed(tokens[t]).logits);
  }

  return {
    name,
    config,
    tokens: Array.from(tokens),
    params: {
      embed: Array.from(params.embed),
      Dx: Array.from(params.Dx),
      Dy: Array.from(params.Dy),
      E: Array.from(params.E),
      head: Array.from(params.head),
    },
    logits_parallel: Array.from(cache.logits.subarray(0, T * config.vocab)),
    logits_recurrent: recurrent,
  };
});

const outDir = join(here, ".cache");
mkdirSync(outDir, { recursive: true });
const outPath = join(outDir, "reference-case.json");
writeFileSync(outPath, JSON.stringify(payload));
console.log(`wrote ${outPath}`);
for (const c of payload) {
  console.log(
    `  case "${c.name}": BDH-GPU(n=${c.config.n}, d=${c.config.d}) L=${c.config.layers} T=${c.tokens.length} vocab=${c.config.vocab}`,
  );
}
