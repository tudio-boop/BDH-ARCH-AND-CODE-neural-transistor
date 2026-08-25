import type { Metadata } from "next";
import Link from "next/link";

import { Section } from "../components/Section";
import { BackendCheck } from "./BackendCheck";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "Backend check",
  description:
    "Internal diagnostics: feed the same bytes through the WebGPU and CPU implementations of the BDH toy and compare the logits.",
};

export default function VerifyPage() {
  return (
    <>
      <div className={styles.header}>
        <div className="shell">
          <p className="label label-ember">Diagnostics</p>
          <h1 className={styles.title}>Backend check</h1>
          <p className="lead" style={{ maxWidth: "62ch" }}>
            The lab has two implementations of the same model. This page feeds
            the same bytes through both and reports how far apart they are.
          </p>
        </div>
      </div>

      <Section index="01" kicker="WebGPU vs CPU" title="Same bytes, both backends">
        <BackendCheck />

        <div className={`grid grid-2 ${styles.notes}`}>
          <div className="card">
            <h3>Why teacher-forced</h3>
            <p className="dim">
              Both backends are fed an identical byte sequence and compared on
              the next-byte logits at every position. Comparing two independent
              generations would measure the sampler instead: with top-k sampling
              a 1e-6 difference in one logit eventually picks a different byte,
              after which the two texts have nothing to do with each other.
            </p>
          </div>
          <div className="card">
            <h3>What counts as agreement</h3>
            <p className="dim">
              The GPU accumulates its dot products in f32; JavaScript numbers are
              f64, so the CPU path accumulates in double precision and rounds
              once. That difference is the floor, and it is what this page
              measures. The threshold for using the GPU at all is set far above
              that floor — it exists to catch a broken driver, not rounding.
            </p>
          </div>
        </div>

        <p className={`footnote ${styles.footer}`}>
          The CPU path is the reference because it is the one checked against the
          PyTorch model: <code>web/scripts/verify_against_bdh_py.py</code>{" "}
          compares both of its forms against <code>bdh.py</code> at identical
          weights. The same comparison shown here also runs from a terminal via{" "}
          <code>web/scripts/check-webgpu.ts</code>. Back to the{" "}
          <Link href="/lab">lab</Link>.
        </p>
      </Section>
    </>
  );
}
