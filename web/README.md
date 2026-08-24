# Baby Dragon Hatchling — web demo

An explainer for [BDH](https://doi.org/10.48550/arXiv.2509.26507) plus a tiny
BDH-GPU that runs in the browser. Next.js App Router, no CSS framework, no web
fonts, no PyTorch, no environment variables.

```bash
npm install
npm run dev      # http://localhost:3000
npm run build
```

Deploying: this directory must be the Vercel project's **Root Directory**. See
the "Deploy on Vercel" section of the [repository README](../README.md).

## Layout

```
app/
  page.tsx              BDH in plain English, and the CPU run card
  architecture/         one BDH-GPU layer, step by step
  lab/                  the in-browser toy: prompt -> neurons -> continuation
lib/bdh/
  model.ts              config, parameters, RoPE   (paper names: E, Dx, Dy)
  linalg.ts             dense float32 kernels, no dependencies
  parallel.ts           token-parallel forward, mirroring bdh.py
  recurrent.ts          the same thing with an explicit rho state
  train.ts              hand-written backward pass and AdamW
  weights.ts            int8 packing for the committed weights
lib/facts.ts            every number the site quotes about the Python model
public/
  bdh-toy-weights.json  trained weights for the browser model
scripts/
  train-toy.ts          trains the browser model (offline, Node)
  gradcheck.ts          backward pass vs finite differences
  check-equivalence.ts  recurrent vs token-parallel
  export-reference-case.ts + verify_against_bdh_py.py   both vs bdh.py
```

## Checks

```bash
npm run typecheck
npm run toy:gradcheck
npm run toy:check
npm run toy:export-ref && python3 scripts/verify_against_bdh_py.py   # needs torch
```

## What the browser model is, and is not

It is BDH-GPU(n=256, d=32), 4 layers, one head, 40,960 parameters, trained on
byte-level tiny Shakespeare by `scripts/train-toy.ts`. The maths matches
`bdh.py` to float32 rounding at identical weights.

It is roughly 618x smaller than the 25,297,152-parameter model `python train.py`
trains at the repository root, and far below the scale at which the paper
measured ~5% activation sparsity or monosemantic synapses. Read its percentages
as this model's, not as the paper's results.
