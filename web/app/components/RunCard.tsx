import {
  CPU_RUN,
  formatCount,
  REPO_DEFAULTS,
  UNIFORM_BYTE_LOSS,
} from "@/lib/facts";
import styles from "./RunCard.module.css";

const SCALE_MIN = 2.5;
const SCALE_MAX = 6;

function positionOf(loss: number): string {
  const clamped = Math.min(SCALE_MAX, Math.max(SCALE_MIN, loss));
  return `${((clamped - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100}%`;
}

export function RunCard() {
  const prompt = REPO_DEFAULTS.promptInTrainPy;
  const continuation = CPU_RUN.sample.startsWith(prompt)
    ? CPU_RUN.sample.slice(prompt.length)
    : CPU_RUN.sample;

  return (
    <div className={`card card-ember ${styles.card}`}>
      <div className={styles.head}>
        <div>
          <p className="label label-ember">Observed run</p>
          <h3 className={styles.title}>
            <code>python train.py</code> on a CPU, stopped at step {CPU_RUN.steps}
          </h3>
        </div>
        <span className="pill">local toy run — not paper scale</span>
      </div>

      <div className={styles.stats}>
        <div>
          <div className="stat-value">{formatCount(CPU_RUN.parameters)}</div>
          <div className="stat-label">parameters</div>
        </div>
        <div>
          <div className="stat-value">{CPU_RUN.steps}</div>
          <div className="stat-label">
            steps of {formatCount(CPU_RUN.stoppedEarlyOf)}
          </div>
        </div>
        <div>
          <div className="stat-value">
            {CPU_RUN.lossStart.toFixed(2)}
            <span className={styles.arrow}> → </span>
            {CPU_RUN.lossEnd.toFixed(2)}
          </div>
          <div className="stat-label">cross-entropy, nats per byte</div>
        </div>
      </div>

      <div className={styles.scale} aria-hidden="true">
        <div className={styles.track}>
          <div
            className={styles.span}
            style={{
              left: positionOf(CPU_RUN.lossEnd),
              right: `calc(100% - ${positionOf(CPU_RUN.lossStart)})`,
            }}
          />
          <div
            className={styles.tick}
            style={{ left: positionOf(UNIFORM_BYTE_LOSS) }}
          />
          <div
            className={`${styles.dot} ${styles.dotStart}`}
            style={{ left: positionOf(CPU_RUN.lossStart) }}
          />
          <div
            className={`${styles.dot} ${styles.dotEnd}`}
            style={{ left: positionOf(CPU_RUN.lossEnd) }}
          />
        </div>
        <div className={styles.scaleLabels}>
          <span>{SCALE_MIN.toFixed(1)}</span>
          <span className={styles.tickLabel}>
            ln 256 = {UNIFORM_BYTE_LOSS.toFixed(2)} (knows nothing)
          </span>
          <span>{SCALE_MAX.toFixed(1)}</span>
        </div>
      </div>

      <p className={styles.explain}>
        The starting loss sits just above <code>ln 256</code>, which is what you
        get from guessing bytes uniformly. Eighty steps later it is at{" "}
        {CPU_RUN.lossEnd.toFixed(2)}, which is enough to have learned letter
        frequencies, spaces and a few common words — and no more. That is exactly
        what the sample looks like.
      </p>

      <p className="label" style={{ marginBottom: "0.6rem" }}>
        Sample after {CPU_RUN.steps} steps — prompt{" "}
        <span className={styles.promptTag}>{JSON.stringify(prompt)}</span>,
        top-k {REPO_DEFAULTS.topK}, {REPO_DEFAULTS.maxNewTokens} new bytes
      </p>
      <pre className="sample">
        <span className={styles.prompt}>{prompt}</span>
        {continuation}
      </pre>

      <div className={styles.configWrap}>
        <p className="label" style={{ marginBottom: "0.75rem" }}>
          Configuration — read from <code>bdh.py</code> and <code>train.py</code>
        </p>
        <div className={styles.config}>
          <ConfigItem label="layers" value={String(REPO_DEFAULTS.layers)} />
          <ConfigItem label="d" value={String(REPO_DEFAULTS.d)} />
          <ConfigItem label="heads" value={String(REPO_DEFAULTS.heads)} />
          <ConfigItem
            label="n"
            value={`${formatCount(REPO_DEFAULTS.neurons)}`}
          />
          <ConfigItem label="vocab" value={`${REPO_DEFAULTS.vocab} bytes`} />
          <ConfigItem label="block" value={String(REPO_DEFAULTS.blockSize)} />
          <ConfigItem label="batch" value={String(REPO_DEFAULTS.batchSize)} />
          <ConfigItem label="lr" value="1e-3 AdamW" />
        </div>
      </div>

      <p className={`footnote ${styles.footnote}`}>
        Four numbers were measured: the parameter count, the step count and the
        two losses. Everything else on this card is configuration copied from the
        source. The run used no GPU, so it covered 80 of the 3,000 steps{" "}
        <code>train.py</code> would have run; the paper&apos;s results are at a
        wholly different scale and are not reproduced here.
      </p>
    </div>
  );
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.configItem}>
      <span className={styles.configLabel}>{label}</span>
      <span className={styles.configValue}>{value}</span>
    </div>
  );
}
