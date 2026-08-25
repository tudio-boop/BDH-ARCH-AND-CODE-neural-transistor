import Link from "next/link";

import { Hero } from "./components/Hero";
import { RunCard } from "./components/RunCard";
import { Section } from "./components/Section";
import styles from "./page.module.css";
import {
  formatCount,
  PAPER,
  REPO_DEFAULTS,
  SCALABLE_PARAMS,
  TOTAL_PARAMS,
} from "@/lib/facts";

export default function HomePage() {
  return (
    <>
      <Hero />

      <Section
        index="01"
        kicker="In plain English"
        title="A Transformer remembers by keeping a list. BDH remembers by changing its wiring."
        id="plain-english"
      >
        <div className={styles.twoCol}>
          <div className="prose">
            <p>
              When a Transformer reads a token, it stores a key and a value
              vector for it, and every later token compares itself against every
              stored key. That store — the KV cache — <em>is</em> the model&apos;s
              working memory, and it gets longer with every token you feed in.
              Attention is a lookup over an ever-growing list of the past.
            </p>
            <p>
              BDH has no such list. It has a fixed population of{" "}
              <strong>n neurons</strong> wired to each other by{" "}
              <strong>synapses</strong>. Reading a token makes a small set of
              those neurons fire. Firing is positive and sparse by construction —
              a ReLU decides who fires, and in a trained model the paper reports
              roughly 5% of neurons active at a time. When one neuron fires just
              after another, the synapse between them is strengthened a little.
              Those synapse strengths are the working memory.
            </p>
            <p>
              This is <strong>Hebbian learning during inference</strong>, which is
              a different thing from training. BDH&apos;s weights are frozen once
              trained. What changes as it reads is the state sitting on the edges
              of the graph — and because the graph has a fixed number of edges,
              the memory has a fixed size no matter how long the input is.
            </p>
            <p>
              The reason to care is in the paper&apos;s title: writing attention
              down as local rules at neurons and synapses turns it into something
              that looks like synaptic plasticity. Same family of scaling
              behaviour as a Transformer, but a mechanism you can inspect. Since
              activations are positive and sparse, you can ask <em>which</em>{" "}
              neurons fired and <em>which</em> synapses strengthened; the paper
              finds individual synapses that strengthen whenever the model is
              handling a particular concept.
            </p>
          </div>

          <div className={styles.compare}>
            <div className="card">
              <p className="label">Transformer</p>
              <h3>Memory is a growing list</h3>
              <ul className="list-marked">
                <li>
                  One key and one value per token, per layer, kept for as long as
                  you need the context.
                </li>
                <li>
                  Attention is a soft lookup: score the current query against
                  every stored key.
                </li>
                <li>
                  Cost per new token grows with how much has been read.
                </li>
                <li>
                  To bound it you have to window, evict or compress old tokens.
                </li>
              </ul>
            </div>
            <div className="card card-ember">
              <p className="label label-ember">BDH</p>
              <h3>Memory is a sheet of synapses</h3>
              <ul className="list-marked">
                <li>
                  A fixed set of synapse strengths, sized when you pick the model
                  — not when you pick the context length.
                </li>
                <li>
                  Reading a token adds one rank-1 update to that sheet; querying
                  it is one matrix-vector product.
                </li>
                <li>
                  Cost per new token is flat, however long the input.
                </li>
                <li>
                  Nothing is evicted. Distant tokens instead drift out of phase
                  under the rotation applied to keys and queries — and the paper
                  notes that over long contexts some damping of stale signal is
                  wanted as well, pairing RoPE with ALiBi. Either way the state
                  never changes size.
                </li>
              </ul>
            </div>
          </div>
        </div>

        <p className={`footnote ${styles.spacedNote}`}>
          Terminology, since both words get used loosely: here{" "}
          <strong>weights</strong> means the trained parameters, which never move
          at inference, and <strong>state</strong> means the synapse values{" "}
          <code>rho</code>, which move on every token and are reset for every new
          input.
        </p>
      </Section>

      <Section
        index="02"
        kicker="Two forms of one model"
        title="BDH is the brain-shaped version. BDH-GPU is the version a GPU will run."
        id="bdh-vs-bdh-gpu"
      >
        <div className="grid grid-2">
          <div className="card">
            <p className="label">BDH</p>
            <h3>A graph of locally-interacting neurons</h3>
            <p className="dim">
              n neuron particles joined by m synapses, updated by rules that only
              ever look at a neuron and its own edges. Excitatory and inhibitory
              circuits, with integrate-and-fire thresholding at each neuron. The
              state is a full synapse matrix <code>sigma</code> of size n x n.
              This is the form the theory is written in, and the form that makes
              the biological claim.
            </p>
          </div>
          <div className="card card-ember">
            <p className="label label-ember">BDH-GPU</p>
            <h3>The same dynamics, as dense tensors</h3>
            <p className="dim">
              Replace wire-by-wire communication with a mean-field broadcast, and
              the n x n state compresses into a low-rank{" "}
              <code>rho = E sigma</code> of size n x d, with d much smaller than
              n. What remains is matmuls, LayerNorm and ReLU, which trains
              token-parallel on a GPU. This is what <code>bdh.py</code>{" "}
              implements and what the toy in the lab runs.
            </p>
          </div>
        </div>

        <div className={styles.tableWrap}>
          <table className="table">
            <thead>
              <tr>
                <th />
                <th>BDH</th>
                <th>BDH-GPU</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="mono dim">state</td>
                <td>
                  <code>sigma</code> in R^(n x n) — one value per synapse
                </td>
                <td>
                  <code>rho = E sigma</code> in R^(n x d) — one d-vector per
                  neuron
                </td>
              </tr>
              <tr>
                <td className="mono dim">parameters</td>
                <td>edge weights of the neuron interaction graphs</td>
                <td>
                  three matrices <code>E</code>, <code>Dx</code>, <code>Dy</code>,
                  shared by every layer
                </td>
              </tr>
              <tr>
                <td className="mono dim">parameter count</td>
                <td>O(nd) for a sparse graph</td>
                <td>(3 + o(1)) &middot; n &middot; d, exactly</td>
              </tr>
              <tr>
                <td className="mono dim">update</td>
                <td>local rules at neurons and synapses</td>
                <td>linear attention + a ReLU-lowrank feed-forward block</td>
              </tr>
              <tr>
                <td className="mono dim">how you scale it</td>
                <td>more neurons, more synapses</td>
                <td>
                  turn one dial: raise n, keep d and everything else fixed
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className={`card ${styles.arithmetic}`}>
          <p className="label label-ember">Where the parameters go</p>
          <p className="dim">
            The paper says a sound choice at the 25M scale is d = 256 and n =
            32,768 — which is exactly what <code>BDHConfig</code> defaults to in
            this repo (4 heads x 8,192 neurons each). So the count is not
            approximate:
          </p>
          <pre className="equation">
            {`3nd  = 3 x 32,768 x 256   = ${formatCount(SCALABLE_PARAMS)}   <- encoder, encoder_v, decoder
     + byte embedding      =         65,536
     + readout             =         65,536
     + unused lm_gate      =            256
                            = ${formatCount(TOTAL_PARAMS)}`}
          </pre>
          <p className="footnote" style={{ marginTop: "0.9rem", marginBottom: 0 }}>
            That total is the parameter count the local CPU run reported below.
            The o(1) term in <em>(3 + o(1))nd</em> is those last three lines:{" "}
            {(((TOTAL_PARAMS - SCALABLE_PARAMS) / TOTAL_PARAMS) * 100).toFixed(2)}
            % of the model.
          </p>
        </div>
      </Section>

      <Section
        index="03"
        kicker="What actually ran"
        title="One CPU, eighty steps, and a model that has just about learned what a space is."
        id="run"
      >
        <RunCard />
      </Section>

      <Section
        index="04"
        kicker="Go deeper"
        title="Read the layer, then run one."
        id="next"
      >
        <div className="grid grid-2">
          <Link href="/architecture" className={`card card-ember ${styles.linkCard}`}>
            <p className="label label-ember">Next</p>
            <h3>How one layer works</h3>
            <p className="dim">
              The six steps of a BDH-GPU layer: lift into neuron space with{" "}
              <code>Dx</code>, take inner products of sparse positive vectors,
              add one rank-1 update to <code>rho</code>, gate through{" "}
              <code>Dy</code>, come back down through <code>E</code>. Each step
              lined up against the line of <code>bdh.py</code> that implements
              it.
            </p>
            <span className={styles.linkCta}>Walk through it →</span>
          </Link>
          <Link href="/lab" className={`card ${styles.linkCard}`}>
            <p className="label">Then</p>
            <h3>The hatchling lab</h3>
            <p className="dim">
              A {formatCount(40960)}-parameter BDH-GPU that runs entirely in this
              tab: type a prompt, watch which neurons fire on each byte, watch{" "}
              <code>rho</code> stay the same size while a KV cache would grow, and
              read the continuation it produces.
            </p>
            <span className={styles.linkCta}>Open the lab →</span>
          </Link>
        </div>

        <p className={`footnote ${styles.spacedNote}`}>
          Everything here follows{" "}
          <a href={PAPER.url} target="_blank" rel="noreferrer">
            {PAPER.title}
          </a>{" "}
          by {PAPER.authors}. Where this site describes code rather than the
          paper, it describes the reference implementation in this repository,
          which uses {REPO_DEFAULTS.layers} layers of shared parameters.
        </p>
      </Section>
    </>
  );
}
