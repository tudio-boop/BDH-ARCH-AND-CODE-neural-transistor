import type { Metadata } from "next";
import Link from "next/link";

import { LayerSteps, type LayerStep } from "../components/LayerSteps";
import { RankOneDiagram } from "../components/RankOneDiagram";
import { Section } from "../components/Section";
import styles from "./page.module.css";
import { formatCount, PAPER, REPO_DEFAULTS } from "@/lib/facts";

export const metadata: Metadata = {
  title: "How one layer works",
  description:
    "A step-by-step walk through one BDH-GPU layer: Dx lifts into neuron space, ReLU makes activations positive and sparse, attention is an inner product between neuron vectors, rho takes one rank-1 update per token, Dy gates, and E brings it back down.",
};

const STEPS: LayerStep[] = [
  {
    n: "1",
    title: "What travels between layers is small",
    body: (
      <>
        <p>
          A BDH-GPU layer takes in a vector of size <strong>d</strong> — here
          256. That is tiny compared with the neuron space it is about to enter,
          and it is the reason the model is cheap: any transformation from
          n-dimensions back to n-dimensions has to squeeze through d on the way.
        </p>
        <p>
          For the first layer this vector is the byte embedding after a
          LayerNorm. For later layers it is the residual handed over by the layer
          before. In the paper&apos;s notation it is{" "}
          <code>v* = LN(E y)</code>, the encoded output of the previous layer.
        </p>
      </>
    ),
    math: `x  in R^d        d = ${REPO_DEFAULTS.d}`,
    code: "x = self.ln(self.embed(idx))",
  },
  {
    n: "2",
    title: "Lift into neuron space, then threshold",
    body: (
      <>
        <p>
          Multiplying by <code>Dx</code> expands the small vector into the
          neuron space: n = {formatCount(REPO_DEFAULTS.neurons)} entries, one per
          neuron. A ReLU then throws away everything negative, so what comes out
          is <strong>positive by construction</strong>.
        </p>
        <p>
          Positivity in a very high dimension is where the sparsity comes from,
          and the paper reports about 5% of entries non-zero in a trained model.
          The neurons left standing are the ones that fire for this token. Nothing
          in the loss asks for this; it is what a ReLU in a big space does.
        </p>
        <p className="footnote">
          Names to keep straight: the paper calls this matrix the{" "}
          <em>decoder</em> <code>Dx</code> because it maps d to n, while{" "}
          <code>bdh.py</code> calls the same tensor <code>encoder</code>. Same
          numbers, opposite convention.
        </p>
      </>
    ),
    math: `x_sparse = ReLU(x @ Dx)      in (R^+)^n     n = ${formatCount(REPO_DEFAULTS.neurons)}`,
    code: "x_latent = x @ self.encoder; x_sparse = F.relu(x_latent)",
  },
  {
    n: "3",
    title: "Attention is an inner product between two sparse positive vectors",
    body: (
      <>
        <p>
          Queries and keys are <em>the same vector</em>: the neuron activation,
          rotated by position. The score between token t and an earlier token is
          just their dot product in neuron space — two firing patterns overlap or
          they do not.
        </p>
        <p>
          There is no softmax. That omission is what makes this{" "}
          <strong>linear attention</strong>, and it is what lets the whole sum be
          folded into a running state in the next step. The mask is strict:{" "}
          <code>tril(..., diagonal=-1)</code>, so a token attends to earlier
          tokens and never to itself. The values are the small d-dimensional
          vectors, not the neuron ones.
        </p>
      </>
    ),
    math: `Q = K = RoPE(x_sparse)        V = x
scores[t, tau] = <Q_t, K_tau>   for tau < t   (no softmax)
a_t = sum over tau < t of scores[t, tau] * V_tau`,
    code: "yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)",
  },
  {
    n: "4",
    title: "Written as a state, that sum is rho — one rank-1 update per token",
    body: (
      <>
        <p>
          Because there is no softmax, the sum over the past can be reorganised.
          Instead of keeping every key and value, keep their accumulated outer
          product. Reading is then a single matrix-vector product, and writing is
          adding one rank-1 term.
        </p>
        <p>
          <code>rho</code> has shape n x d: <strong>one d-vector per neuron</strong>
          , which the paper reads as the state of neuron i as a particle. Since
          the key is sparse and positive, a silent neuron contributes an
          all-zeros row — a token only touches the synapses of neurons that fired
          for it.
        </p>
        <p>
          This is the whole memory. Its size is n &middot; d per layer, fixed
          when you choose the model, whether you feed it ten bytes or ten
          thousand.
        </p>
      </>
    ),
    math: `read    a_t  = rho_t^T q_t
write   rho_(t+1) = rho_t + k_t (outer) v_t
size    n * d per layer, constant in sequence length`,
    code: "scores = (QR @ KR.mT).tril(diagonal=-1); return scores @ V",
  },
  {
    n: "5",
    title: "Gate the neurons that fired with what memory returned",
    body: (
      <>
        <p>
          The read-out is normalised, pushed back up into neuron space by{" "}
          <code>Dy</code>, and thresholded again. Then comes the move that gives
          the block its name — a <strong>multiplicative</strong> gate:
          elementwise product with the neurons that fired in step 2.
        </p>
        <p>
          So a neuron contributes only if it both fired for this token{" "}
          <em>and</em> is picked out by what attention retrieved. Two ReLU masks
          multiplied together make <code>y</code> sparser than either one alone.
          This <code>y</code> is the vector the paper measures at roughly 5%
          non-zero.
        </p>
      </>
    ),
    math: `gate = ReLU(LN(a) @ Dy)
y    = gate * x_sparse        (elementwise, in neuron space)`,
    code: "y_sparse = F.relu(yKV @ self.encoder_v); xy_sparse = x_sparse * y_sparse",
  },
  {
    n: "6",
    title: "Come back down through E — and do it again with the same weights",
    body: (
      <>
        <p>
          <code>E</code> projects the neuron space back down to d, a LayerNorm
          tidies it, and it is added to the residual. That output is the next
          layer&apos;s input.
        </p>
        <p>
          The part worth pausing on: all {REPO_DEFAULTS.layers} layers use{" "}
          <strong>the same</strong> <code>E</code>, <code>Dx</code> and{" "}
          <code>Dy</code>. There are no per-layer parameters, which is why the
          count is 3nd no matter how deep you stack it, and why depth here is
          closer to &quot;more reasoning time&quot; than &quot;more model&quot;.
        </p>
      </>
    ),
    math: `x <- LN(x + LN(y @ E))
repeat for all L layers, sharing E, Dx, Dy
logits = x @ lm_head`,
    code: "yMLP = xy_sparse... @ self.decoder; x = self.ln(x + self.ln(yMLP))",
  },
];

export default function ArchitecturePage() {
  return (
    <>
      <div className={styles.header}>
        <div className="shell">
          <p className="label label-ember">The mechanism</p>
          <h1 className={styles.title}>How one layer works</h1>
          <p className="lead" style={{ maxWidth: "62ch" }}>
            Six steps, in the order the code runs them. Two names to hold on to:{" "}
            <strong>d</strong> is the small dimension that travels between layers
            ({REPO_DEFAULTS.d} here), and <strong>n</strong> is the number of
            neurons ({formatCount(REPO_DEFAULTS.neurons)} here). Everything
            interesting happens in n; everything cheap happens in d.
          </p>
        </div>
      </div>

      <Section index="01" kicker="Step by step" title="One BDH-GPU layer">
        <LayerSteps steps={STEPS} />
      </Section>

      <Section
        index="02"
        kicker="The state"
        title="What a rank-1 update actually looks like"
      >
        <RankOneDiagram />

        <div className={`grid grid-2 ${styles.stateGrid}`}>
          <div className="card">
            <p className="label">Cost of memory in BDH-GPU</p>
            <h3>n &middot; d per layer, and that is the end of it</h3>
            <p className="dim">
              With this repo&apos;s defaults that is{" "}
              {formatCount(REPO_DEFAULTS.neurons * REPO_DEFAULTS.d)} numbers per
              layer. Reading is one matrix-vector product; writing is one outer
              product. Both are the same amount of work on token 5 as on token
              5,000.
            </p>
          </div>
          <div className="card">
            <p className="label">Cost of memory in a KV cache</p>
            <h3>2 &middot; T &middot; d per layer, and T keeps rising</h3>
            <p className="dim">
              At the same d, a Transformer&apos;s cache passes BDH-GPU&apos;s
              state size at T = n / 2 tokens — and unlike the state, it keeps
              growing after that. The lab draws this line as you generate.
            </p>
          </div>
        </div>
      </Section>

      <Section
        index="03"
        kicker="Reading the code"
        title="The same tensors, under three different names"
      >
        <p className="prose" style={{ maxWidth: "72ch" }}>
          The paper and the reference implementation disagree about naming, which
          makes <code>bdh.py</code> harder to read than it needs to be. This is
          the mapping the demo uses everywhere, including in its TypeScript port.
        </p>
        <div className={styles.tableWrap}>
          <table className="table">
            <thead>
              <tr>
                <th>Paper</th>
                <th>bdh.py</th>
                <th>Shape (defaults)</th>
                <th>Job</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="mono">Dx</td>
                <td className="mono dim">encoder</td>
                <td className="num dim">4 x 256 x 8192</td>
                <td>d to neuron space, giving queries and keys</td>
              </tr>
              <tr>
                <td className="mono">Dy</td>
                <td className="mono dim">encoder_v</td>
                <td className="num dim">4 x 256 x 8192</td>
                <td>d to neuron space, giving the gate</td>
              </tr>
              <tr>
                <td className="mono">E</td>
                <td className="mono dim">decoder</td>
                <td className="num dim">32768 x 256</td>
                <td>neuron space back down to d</td>
              </tr>
              <tr>
                <td className="mono">rho</td>
                <td className="mono dim">implicit in scores @ V</td>
                <td className="num dim">32768 x 256 per layer</td>
                <td>the working memory</td>
              </tr>
              <tr>
                <td className="mono">U</td>
                <td className="mono dim">get_freqs, rope</td>
                <td className="num dim">—</td>
                <td>
                  applies position to keys and queries — rotation here, though
                  the paper also allows damping such as ALiBi
                </td>
              </tr>
              <tr>
                <td className="mono">x in (R+)^n</td>
                <td className="mono dim">x_sparse</td>
                <td className="num dim">32768</td>
                <td>which neurons fired</td>
              </tr>
              <tr>
                <td className="mono">y</td>
                <td className="mono dim">xy_sparse</td>
                <td className="num dim">32768</td>
                <td>the gated, sparse, positive output</td>
              </tr>
              <tr>
                <td className="mono">v* = LN(E y)</td>
                <td className="mono dim">x</td>
                <td className="num dim">256</td>
                <td>the small vector between layers</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className={`card ${styles.caveats}`}>
          <p className="label label-ember">Two honest differences</p>
          <ul className="list-marked">
            <li>
              <strong>Where the residual lives.</strong> Equation 8 of the paper
              accumulates the residual in the n-dimensional neuron space, while{" "}
              <code>bdh.py</code> accumulates it in the d-dimensional space and
              recomputes the neuron vector from scratch in each layer. This page,
              and the toy in the lab, follow <code>bdh.py</code> — the code that
              is actually in this repository.
            </li>
            <li>
              <strong>Heads.</strong> The equations above are the single-head
              case. <code>bdh.py</code> splits the {formatCount(REPO_DEFAULTS.neurons)}{" "}
              neurons into {REPO_DEFAULTS.heads} groups of{" "}
              {formatCount(REPO_DEFAULTS.neuronsPerHead)} and runs the score
              computation inside each group, which leaves the total state size at
              n &middot; d either way. The browser toy uses one head.
            </li>
          </ul>
        </div>

        <p className={`footnote ${styles.footer}`}>
          Equation numbering and the ~5% sparsity figure are from{" "}
          <a href={PAPER.url} target="_blank" rel="noreferrer">
            {PAPER.title}
          </a>
          . Ready to watch it run?{" "}
          <Link href="/lab">Open the lab</Link>.
        </p>
      </Section>
    </>
  );
}
