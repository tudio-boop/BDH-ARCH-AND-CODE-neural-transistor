import type { Metadata } from "next";
import Link from "next/link";

import { Section } from "../components/Section";
import styles from "./page.module.css";
import { ToyLab } from "./ToyLab";
import { TOY_CONFIG, paramCount } from "@/lib/bdh/model";
import { formatCount } from "@/lib/facts";

export const metadata: Metadata = {
  title: "The hatchling lab",
  description:
    "A 40,960-parameter BDH-GPU running entirely in your browser: watch which neurons fire on each byte of your prompt, watch the rho state stay a fixed size while a KV cache would grow, and read what it writes next.",
};

export default function LabPage() {
  const toyParams = paramCount(TOY_CONFIG);

  return (
    <>
      <div className={styles.header}>
        <div className="shell">
          <p className="label label-ember">Hands on</p>
          <h1 className={styles.title}>The hatchling lab</h1>
          <p className="lead" style={{ maxWidth: "64ch" }}>
            A real BDH-GPU — {formatCount(toyParams)} parameters of it — running
            in this tab in plain JavaScript. No PyTorch, no server, no API call.
            It reads your prompt one byte at a time, and you can watch which
            neurons fire for each one.
          </p>
          <p className={`footnote ${styles.note}`}>
            The maths is the same as{" "}
            <Link href="/architecture">the layer you just read about</Link>: the
            same three matrices, the same rank-1 update to <code>rho</code>. It is
            checked against the PyTorch model in <code>bdh.py</code> to within
            float32 rounding, at identical weights, by{" "}
            <code>web/scripts/verify_against_bdh_py.py</code>.
          </p>
        </div>
      </div>

      <section className="section" style={{ borderTop: "none", paddingTop: 0 }}>
        <div className="shell">
          <ToyLab />
        </div>
      </section>

      <Section
        index="—"
        kicker="How to read it"
        title="What to look for while it runs"
      >
        <div className="grid grid-3">
          <div className="card">
            <h3>Sparsity, not activity</h3>
            <p className="dim">
              Most cells stay dark. Switch between <code>x</code>,{" "}
              <code>gate</code> and <code>y</code>: because <code>y</code> is the
              product of two ReLU&apos;d vectors, it is always the sparsest of the
              three. That multiplication is the whole reason BDH activations end
              up sparse without anything in the loss asking for it.
            </p>
            <p className="dim">
              The first byte is a special case worth catching: there is nothing
              in <code>rho</code> yet, so the gate is exactly zero and{" "}
              <code>y</code> is empty. The first byte can only write.
            </p>
          </div>
          <div className="card">
            <h3>The state does not grow</h3>
            <p className="dim">
              The <code>rho</code> bar never moves, however many bytes you
              generate. The KV cache bar next to it keeps climbing. That is the
              architectural claim, in two bars: memory is a property of the model
              you chose, not of the conversation you are having.
            </p>
          </div>
          <div className="card">
            <h3>Different neurons for different bytes</h3>
            <p className="dim">
              Watch the grids as it reads a space, then a letter. The firing
              pattern changes shape. At this size the patterns are not
              interpretable, but the paper&apos;s monosemanticity results are
              measured on exactly this object, several orders of magnitude larger.
            </p>
          </div>
        </div>

        <p className={`footnote ${styles.footer}`}>
          Want the real thing instead? <code>pip install -r requirements.txt</code>{" "}
          then <code>python train.py</code> at the repo root trains the
          25,297,152-parameter model this site is about. This lab exists because
          that will not run in a serverless function, not as a replacement for
          it.
        </p>
      </Section>
    </>
  );
}
