## Baby Dragon Hatchling
This repository contains source code from the paper: Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz, _"The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"_, [link](https://doi.org/10.48550/arXiv.2509.26507).

## Architecture
<img src="figs/architecture.png" width="600"/> 

## Relation to Tranformers
<img src="figs/vocab.png" width="600"/> 

## Scaling laws
<img src="figs/bdh_scaling.png" width="600"/> 

## Abstract:
The relationship between computing systems and the brain has served as motivation for pioneering theoreticians since John von Neumann and Alan Turing. 
Uniform, scale-free biological networks, such as the brain, have powerful properties, including generalizing over time, which is the main barrier for Machine Learning on the path to Universal Reasoning Models.

We introduce `Dragon Hatchling' (BDH), a new Large Language Model architecture based on a scale-free biologically inspired network of $n$ locally-interacting neuron particles. BDH couples strong theoretical foundations and inherent interpretability without sacrificing Transformer-like performance.

BDH is a practical, performant state-of-the-art 
attention-based state space sequence learning architecture. 
In addition to being a graph model, BDH admits a GPU-friendly formulation.
It exhibits Transformer-like scaling laws: we find empirically that BDH rivals GPT2-architecture Transformer performance on language and translation tasks, at the same number of parameters (10M to 1B), for the same training data.

BDH provides theoretical foundations for understanding model behavior in the limit of large size and reasoning time. 
Our results, formalized as a chain of reductions of expressiveness in the framework of computational Complexity Theory and Distributed Computing, and combined with findings on the BDH model, show a macro-to-micro correspondence of function between the general attention mechanisms in state-of-the-art Language Models, and attention mechanisms observed in the brain. These attention mechanisms formally converge as closed-form local graph dynamics at neurons and synapses: _the equations of reasoning_.

BDH can be represented as a brain model. It contains $n$ neurons, organized as an excitatory circuit and an inhibitory circuit with integrate-and-fire thresholding of input signals at neurons. The working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons, at potentiation scales of minutes for the brain (up to hundreds of tokens). We confirm empirically that specific, individual synapses strengthen connection whenever BDH hears or reasons about a specific concept while processing language inputs. The neuron interaction network of BDH is a graph of high modularity with heavy-tailed degree distribution. The BDH model is biologically plausible, explaining one possible mechanism which human neurons could use to achieve speech.

BDH is designed for interpretability. Activation vectors of BDH are sparse and positive. We demonstrate monosemanticity in BDH on language tasks, including representation of concept abstractions, which happens even for small models, below 100M-parameter scale. Interpretability of state, which goes beyond interpretability of neurons and model parameters, is an inherent feature of the BDH architecture. 

We believe BDH opens the door to a new theory of _Thermodynamic Limit_ behavior for language and reasoning models, with the ultimate goal of Probably Approximately Correct (PAC)-like bounds for generalization of reasoning over time.

## Running the code

To train and sample from the BDH model on a toy language modeling task please do:
1. `pip install -r requirements.txt`
2. `python train.py`

## Web demo (`web/`)

`web/` holds a self-contained Next.js (App Router) site that explains BDH and
runs a tiny BDH-GPU in the browser. It is independent of the Python code above:
it has no PyTorch dependency, adds nothing to `requirements.txt`, and does not
change how `python train.py` behaves.

Three pages:

| Page | What it is |
| --- | --- |
| `/` | BDH in plain English — synapse graph versus KV cache, BDH versus BDH-GPU, and a card with the numbers from a local CPU training run |
| `/architecture` | One BDH-GPU layer, step by step: `Dx` into neuron space, sparse positive activations, the rank-1 update to `rho`, the `Dy` gate, back down through `E` |
| `/lab` | A 40,960-parameter BDH-GPU running in the browser, on the GPU via WebGPU where available: type a prompt, watch which neurons fire on each byte, and read the continuation |
| `/verify` | Diagnostics: feeds the same bytes through both backends and reports how far apart they are |

The browser model is a real trained model, not a mock. `web/lib/bdh/` is a hand
port of `bdh.py` (with `n_head=1` and `dropout=0`) implemented twice: the
token-parallel triangular form that `bdh.py` uses, and the recurrent form that
carries an explicit `rho` state, which is what makes the "no KV cache" claim
demonstrable.

### Two backends

`/lab` runs the recurrent form on **WebGPU** when the browser exposes it, and on
the **CPU in TypeScript** when it does not. There is nothing to switch on: the
lab feeds a short prompt through both at startup and only uses the GPU if its
logits agree with the CPU reference. `web/lib/bdh/webgpu/shader.ts` does the
whole token step — every layer, then the readout — in a single workgroup, so
each byte is one dispatch; at n = 256 that shape matters far more than
arithmetic throughput, since a byte is only about 170,000 multiply-adds.

### Checks

```bash
cd web
npm install
npm run toy:gradcheck   # backward pass vs finite differences
npm run toy:check       # recurrent form vs token-parallel form
npm run toy:export-ref && python3 scripts/verify_against_bdh_py.py   # both vs bdh.py, same weights
npm run toy:check-webgpu   # WebGPU vs CPU, needs Deno (see below)
```

The PyTorch comparison needs `torch`, imports `bdh.py` read-only, and agrees
with the reference implementation to within float32 rounding (~5e-7 relative).

The WebGPU comparison runs under [Deno](https://deno.com), which ships a WebGPU
implementation, so the GPU path can be checked from a terminal without a browser
and without adding a native dependency here. Deno is not needed to build, deploy
or use the site — opening `/verify` in a browser runs the same comparison. On a
Linux box with no GPU, point Deno at a software Vulkan device:

```bash
VK_ICD_FILENAMES=/opt/google/chrome/vk_swiftshader_icd.json npm run toy:check-webgpu
```

### Run the site locally

```bash
cd web
npm install
npm run dev        # http://localhost:3000
npm run build      # production build
```

### Deploy on Vercel

> **This demo is internal.** It is meant to be reachable only by the team that
> owns the Vercel project, so keep Deployment Protection switched on and do not
> add a public domain. The app sets `noindex, nofollow` and ships a
> `robots.txt` that disallows every crawler, but protection on the Vercel
> project is what actually keeps it private. Note that on a Pro plan Vercel
> Authentication covers previews and production *deployment URLs* but not the
> production domain itself, so a project whose `*.vercel.app` production alias
> is assigned is publicly reachable unless you also remove that alias, pause the
> project, or add the deployment-protection add-on that covers production.

The Vercel project's **Root Directory must be `web`**, because the repository
root is a Python project.

1. Go to [vercel.com/new](https://vercel.com/new) and import this repository (or
   your fork of it).
2. On the import screen, expand **Root Directory** and set it to `web`.
3. Leave everything else alone. Vercel reads `web/vercel.json`, which pins the
   Next.js preset, and the default install and build commands are correct. There
   are no environment variables or secrets to add.
4. Deploy. Every route is prerendered as static content.

From the CLI the equivalent is to run Vercel from inside the subdirectory:

```bash
cd web
npx vercel          # preview deployment
npx vercel --prod   # production deployment
```

Root Directory is a **project setting**, not a `vercel.json` field — there is no
`rootDirectory` key in [the `vercel.json`
schema](https://openapi.vercel.sh/vercel.json) — so it cannot be committed to
the repository root, and step 3 is the one manual step. If it is skipped, the
build fails at framework detection because there is no `package.json` at the
repository root.

### Regenerating the browser model's weights

The trained weights are committed at `web/public/bdh-toy-weights.json` (int8
with a scale per row, ~70 KB). To retrain them:

```bash
cd web
npm run toy:train -- --steps 4000 --batch 10 --block 128
```

That downloads the same tiny Shakespeare corpus `train.py` uses, trains
BDH-GPU(n=256, d=32) with 4 layers using AdamW at the same learning rate, prints
samples as it goes, and rewrites the weights file along with the training
metadata the `/lab` page displays.

## Acknowledgements
We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shapespeare dataset used in this demonstration.
