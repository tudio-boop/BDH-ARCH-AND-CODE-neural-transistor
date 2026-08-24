import { PAPER, UPSTREAM_REPO } from "@/lib/facts";
import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  return (
    <footer className={styles.footer}>
      <div className={`shell ${styles.inner}`}>
        <div className={styles.col}>
          <p className="label">The paper</p>
          <p className={styles.body}>
            <a href={PAPER.url} target="_blank" rel="noreferrer">
              {PAPER.title}
            </a>
            <br />
            <span className="dim">{PAPER.authors}</span>
            <br />
            <span className="dim">
              {PAPER.org} — arXiv:{PAPER.arxiv}
            </span>
          </p>
        </div>

        <div className={styles.col}>
          <p className="label">This site</p>
          <p className={styles.body}>
            An explainer and an in-browser toy, built on top of the reference
            implementation in <code>bdh.py</code>. The Python trainer is
            untouched: <code>python train.py</code> still trains the 25M model.
            Reference code:{" "}
            <a href={UPSTREAM_REPO} target="_blank" rel="noreferrer">
              pathwaycom/bdh
            </a>
            .
          </p>
        </div>

        <div className={styles.col}>
          <p className="label">Honest limits</p>
          <p className={styles.body}>
            Nothing here reproduces a paper-scale result. The numbers on the run
            card come from one local CPU run that was stopped after 80 steps, and
            the browser model has 40,960 parameters.
          </p>
        </div>
      </div>
    </footer>
  );
}
