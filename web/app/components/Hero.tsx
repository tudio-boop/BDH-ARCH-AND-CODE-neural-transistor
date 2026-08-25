import Link from "next/link";

import { formatCount, PAPER, REPO_DEFAULTS, TOTAL_PARAMS } from "@/lib/facts";
import { HatchlingEgg } from "./HatchlingEgg";
import styles from "./Hero.module.css";

export function Hero() {
  return (
    <div className={styles.hero}>
      <div className={`shell ${styles.inner}`}>
        <div className={styles.copy}>
          <p className="label label-ember">
            arXiv:{PAPER.arxiv} — {PAPER.org}
          </p>
          <h1 className={styles.title}>
            Baby Dragon
            <br />
            <em>Hatchling</em>
          </h1>
          <p className="lead">
            A language model that keeps its working memory in{" "}
            <strong>synapses</strong> rather than a KV cache. Reading a token
            makes a sparse set of neurons fire; the connections between the ones
            that fired together get stronger, and that is the entire memory of
            the conversation.
          </p>
          <div className={styles.pills}>
            <span className="pill pill-ember">
              <span className="pill-dot" />n = {formatCount(REPO_DEFAULTS.neurons)} neurons
            </span>
            <span className="pill">d = {REPO_DEFAULTS.d}</span>
            <span className="pill">{formatCount(TOTAL_PARAMS)} params</span>
            <span className="pill">fixed-size state</span>
          </div>
          <div className={styles.actions}>
            <Link className="button" href="/lab">
              Hatch one in your browser
            </Link>
            <Link className="button button-ghost" href="/architecture">
              Walk through a layer
            </Link>
          </div>
        </div>

        <div className={styles.art}>
          <HatchlingEgg />
          <p className={styles.caption}>
            n neurons, sparsely firing. The lit edges are the working memory.
          </p>
        </div>
      </div>
    </div>
  );
}
