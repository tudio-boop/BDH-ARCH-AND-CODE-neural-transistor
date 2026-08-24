/**
 * Every number this site shows about the PyTorch model, in one place.
 *
 * Two kinds of facts live here, kept apart on purpose:
 *
 *  - `REPO_DEFAULTS` and `PARAMETER_BREAKDOWN` are read straight off the
 *    committed source (bdh.py, train.py). They are arithmetic, not results.
 *  - `CPU_RUN` is what one actual local CPU run of `python train.py` printed.
 *    It is a toy run that was stopped early. Nothing here is extrapolated, and
 *    no metric appears that was not observed.
 */

export const PAPER = {
  title: "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain",
  authors:
    "Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, Michał Bartoszkiewicz",
  arxiv: "2509.26507",
  url: "https://doi.org/10.48550/arXiv.2509.26507",
  org: "Pathway",
} as const;

export const UPSTREAM_REPO = "https://github.com/pathwaycom/bdh";

/** bdh.BDHConfig defaults, plus the training constants at the top of train.py. */
export const REPO_DEFAULTS = {
  layers: 6,
  d: 256,
  heads: 4,
  mlpMultiplier: 128,
  /** N = mlp_internal_dim_multiplier * n_embd // n_head */
  neuronsPerHead: 8192,
  /** n = heads * N: the paper's neuron dimension */
  neurons: 32768,
  vocab: 256,
  dropout: 0.1,
  blockSize: 512,
  batchSize: 32,
  maxIters: 3000,
  learningRate: 1e-3,
  weightDecay: 0.1,
  promptInTrainPy: "To be or ",
  maxNewTokens: 100,
  topK: 3,
} as const;

/**
 * Where the 25,297,152 parameters sit. The paper's claim is that a BDH-GPU has
 * (3 + o(1))nd parameters; with this repo's defaults the 3nd term is exact and
 * the o(1) term is the byte embedding plus the readout.
 */
export const PARAMETER_BREAKDOWN = [
  {
    name: "encoder",
    paperName: "Dx",
    shape: "4 x 256 x 8192",
    count: 8_388_608,
    role: "lifts the residual into neuron space",
  },
  {
    name: "encoder_v",
    paperName: "Dy",
    shape: "4 x 256 x 8192",
    count: 8_388_608,
    role: "lifts the attention read-out into neuron space",
  },
  {
    name: "decoder",
    paperName: "E",
    shape: "32768 x 256",
    count: 8_388_608,
    role: "projects the neuron space back down",
  },
  {
    name: "embed.weight",
    paperName: "f_e",
    shape: "256 x 256",
    count: 65_536,
    role: "byte embedding",
  },
  {
    name: "lm_head",
    paperName: "f_d",
    shape: "256 x 256",
    count: 65_536,
    role: "logits over the 256 byte values",
  },
  {
    name: "lm_gate",
    paperName: "—",
    shape: "256 x 1",
    count: 256,
    role: "allocated by bdh.py but unused in forward()",
  },
] as const;

export const TOTAL_PARAMS = 25_297_152;
/** 3nd with n = 32768, d = 256 */
export const SCALABLE_PARAMS = 25_165_824;

/**
 * One local CPU run of this repo's `python train.py`, interrupted at step 80.
 * These four numbers are the whole of what was measured.
 */
export const CPU_RUN = {
  device: "CPU",
  parameters: TOTAL_PARAMS,
  steps: 80,
  lossStart: 5.68,
  lossEnd: 2.87,
  stoppedEarlyOf: REPO_DEFAULTS.maxIters,
  sample:
    "To be or to the t ton sese t s t n son ne s n the s t t n n s thet s se s torethes se nethe s s sondo thethes",
} as const;

/** ln(256): what the loss is before the model knows anything at all. */
export const UNIFORM_BYTE_LOSS = Math.log(256);

export function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}
