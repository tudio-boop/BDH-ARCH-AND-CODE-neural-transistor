import type { ReactNode } from "react";

import styles from "./LayerSteps.module.css";

export interface LayerStep {
  n: string;
  title: string;
  body: ReactNode;
  math: string;
  code: string;
}

export function LayerSteps({ steps }: { steps: LayerStep[] }) {
  return (
    <ol className={styles.list}>
      {steps.map((step) => (
        <li key={step.n} className={styles.step}>
          <div className={styles.marker} aria-hidden="true">
            <span className={styles.markerNumber}>{step.n}</span>
            <span className={styles.markerLine} />
          </div>
          <div className={styles.content}>
            <h3 className={styles.title}>{step.title}</h3>
            <div className={`prose ${styles.body}`}>{step.body}</div>
            <pre className="equation">{step.math}</pre>
            <p className={styles.code}>
              <span className={styles.codeLabel}>bdh.py</span>
              <code>{step.code}</code>
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}
