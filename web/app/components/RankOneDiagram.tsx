import styles from "./RankOneDiagram.module.css";

/**
 * The rank-1 update, drawn as a multiplication table: a sparse positive column
 * (the neurons that fired) times a dense low-rank row (the residual), added
 * into rho. Rows belonging to silent neurons stay exactly zero, which is the
 * "only strengthened synapses are written" claim made visible.
 */

const ROWS = 16;
const COLS = 6;
const FIRING = [2, 6, 11];
/** arbitrary but fixed, so the picture is stable across renders */
const K_VALUES = [0.86, 0.42, 0.61];
const V_VALUES = [0.7, -0.35, 0.95, 0.15, -0.6, 0.45];

const CELL_W = 26;
const CELL_H = 15;
const GAP = 2;
const GRID_X = 132;
const GRID_Y = 74;

function emberFill(magnitude: number): string {
  const a = Math.min(1, Math.abs(magnitude));
  return `rgba(255, ${Math.round(122 + 60 * a)}, ${Math.round(47 + 30 * a)}, ${0.18 + 0.72 * a})`;
}

/** v is signed, unlike everything in neuron space, so negatives get a cool tone. */
function signedFill(value: number): string {
  if (value >= 0) return emberFill(value);
  const a = Math.min(1, Math.abs(value));
  return `rgba(139, 160, 168, ${0.16 + 0.6 * a})`;
}

export function RankOneDiagram() {
  const gridW = COLS * (CELL_W + GAP) - GAP;
  const gridH = ROWS * (CELL_H + GAP) - GAP;

  return (
    <figure className={styles.figure}>
      <div className={styles.scroller}>
        <svg
          viewBox={`0 0 640 ${GRID_Y + gridH + 46}`}
          className={styles.svg}
          role="img"
          aria-label="A sparse column vector times a dense row vector, forming a matrix whose only non-zero rows belong to the neurons that fired, added into the state rho"
        >
          {/* v: the d-dimensional row */}
          <text x={GRID_X} y={GRID_Y - 34} className={styles.axisLabel}>
            v = x &nbsp;(d entries, dense)
          </text>
          {V_VALUES.map((value, j) => (
            <rect
              key={`v-${j}`}
              x={GRID_X + j * (CELL_W + GAP)}
              y={GRID_Y - 26}
              width={CELL_W}
              height={CELL_H}
              rx="1.5"
              fill={signedFill(value)}
              stroke="rgba(255,169,94,0.22)"
              strokeWidth="0.6"
            />
          ))}

          {/* k: the n-dimensional sparse positive column */}
          <text
            className={styles.axisLabel}
            transform={`translate(${GRID_X - 92} ${GRID_Y + gridH / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            k = RoPE(x⁺) &nbsp;(n entries, sparse)
          </text>
          {Array.from({ length: ROWS }, (_, i) => {
            const firingIndex = FIRING.indexOf(i);
            const lit = firingIndex >= 0;
            return (
              <rect
                key={`k-${i}`}
                x={GRID_X - 62}
                y={GRID_Y + i * (CELL_H + GAP)}
                width={CELL_W}
                height={CELL_H}
                rx="1.5"
                fill={lit ? emberFill(K_VALUES[firingIndex]) : "rgba(244,232,216,0.045)"}
                stroke={lit ? "rgba(255,169,94,0.45)" : "rgba(244,232,216,0.07)"}
                strokeWidth="0.6"
              />
            );
          })}

          <text
            x={GRID_X - 24}
            y={GRID_Y + gridH / 2 + 4}
            className={styles.operator}
          >
            ⊗
          </text>

          {/* the outer product */}
          {Array.from({ length: ROWS }, (_, i) => {
            const firingIndex = FIRING.indexOf(i);
            return Array.from({ length: COLS }, (_, j) => {
              const lit = firingIndex >= 0;
              const value = lit ? K_VALUES[firingIndex] * V_VALUES[j] : 0;
              return (
                <rect
                  key={`p-${i}-${j}`}
                  x={GRID_X + j * (CELL_W + GAP)}
                  y={GRID_Y + i * (CELL_H + GAP)}
                  width={CELL_W}
                  height={CELL_H}
                  rx="1.5"
                  fill={lit ? signedFill(value) : "rgba(244,232,216,0.035)"}
                  stroke={lit ? "rgba(255,169,94,0.24)" : "rgba(244,232,216,0.055)"}
                  strokeWidth="0.6"
                />
              );
            });
          })}

          {/* into rho */}
          <text
            x={GRID_X + gridW + 30}
            y={GRID_Y + gridH / 2 - 6}
            className={styles.operator}
          >
            +
          </text>
          <text
            x={GRID_X + gridW + 62}
            y={GRID_Y - 26 + CELL_H}
            className={styles.axisLabel}
          >
            rho (n x d, per layer)
          </text>
          <rect
            x={GRID_X + gridW + 58}
            y={GRID_Y - 4}
            width={gridW + 8}
            height={gridH + 8}
            rx="3"
            fill="rgba(147,41,15,0.16)"
            stroke="var(--vein)"
            strokeWidth="1"
          />
          {Array.from({ length: ROWS * COLS }, (_, index) => {
            const i = Math.floor(index / COLS);
            const j = index % COLS;
            // a plausible accumulated state: most rows already carry something
            const seeded = (((i * 7 + j * 13) % 11) / 11) * 2 - 0.55;
            return (
              <rect
                key={`rho-${index}`}
                x={GRID_X + gridW + 62 + j * (CELL_W + GAP)}
                y={GRID_Y + i * (CELL_H + GAP)}
                width={CELL_W}
                height={CELL_H}
                rx="1.5"
                fill={signedFill(seeded * 0.6)}
                opacity={0.85}
              />
            );
          })}

          <text
            x={GRID_X + gridW / 2}
            y={GRID_Y + gridH + 26}
            className={styles.caption}
            textAnchor="middle"
          >
            rank 1 per token
          </text>
          <text
            x={GRID_X + gridW + 62 + gridW / 2}
            y={GRID_Y + gridH + 26}
            className={styles.caption}
            textAnchor="middle"
          >
            always this size
          </text>
        </svg>
      </div>
      <figcaption className="footnote">
        Silent neurons contribute an exactly-zero row, so a token only writes to
        the synapses of the neurons that fired for it. The picture uses 16 rows
        so it fits on a screen; the real thing has n = 32,768 of them and only a
        few percent are lit.
      </figcaption>
    </figure>
  );
}
