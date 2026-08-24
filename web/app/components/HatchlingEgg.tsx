import styles from "./HatchlingEgg.module.css";

/**
 * The hero glyph: an egg with a scale-free neuron graph glowing inside it, a
 * few synapses lit brighter than the rest. It is the picture the paper paints —
 * working memory as edge weights, not as a list of past tokens.
 */

const NODES: { x: number; y: number; r: number }[] = [
  { x: 160, y: 112, r: 4.5 },
  { x: 112, y: 146, r: 3.4 },
  { x: 206, y: 142, r: 5.6 },
  { x: 88, y: 196, r: 3.1 },
  { x: 143, y: 178, r: 6.6 },
  { x: 196, y: 196, r: 3.8 },
  { x: 233, y: 188, r: 3.2 },
  { x: 113, y: 241, r: 4.9 },
  { x: 162, y: 232, r: 7.4 },
  { x: 209, y: 246, r: 3.5 },
  { x: 132, y: 293, r: 3.9 },
  { x: 184, y: 289, r: 4.6 },
  { x: 157, y: 326, r: 3.2 },
];

/** [from, to, lit] — `lit` synapses are the ones carrying state right now. */
const EDGES: [number, number, boolean][] = [
  [0, 1, false],
  [0, 2, true],
  [0, 4, false],
  [1, 3, false],
  [1, 4, true],
  [2, 4, false],
  [2, 5, false],
  [2, 6, false],
  [3, 7, false],
  [4, 5, false],
  [4, 8, true],
  [5, 8, false],
  [5, 9, false],
  [6, 9, false],
  [7, 8, true],
  [7, 10, false],
  [8, 9, false],
  [8, 11, true],
  [8, 10, false],
  [9, 11, false],
  [10, 11, false],
  [10, 12, false],
  [11, 12, true],
  [4, 7, false],
  [2, 8, false],
];

export function HatchlingEgg({ className }: { className?: string }) {
  return (
    <svg
      className={[styles.egg, className].filter(Boolean).join(" ")}
      viewBox="0 0 320 400"
      role="img"
      aria-label="An egg containing a network of neurons, with several synapses glowing"
    >
      <defs>
        <radialGradient id="eggFill" cx="50%" cy="34%" r="72%">
          <stop offset="0%" stopColor="#2a1a11" />
          <stop offset="58%" stopColor="#170f0a" />
          <stop offset="100%" stopColor="#0c0705" />
        </radialGradient>
        <radialGradient id="eggGlow" cx="50%" cy="62%" r="58%">
          <stop offset="0%" stopColor="rgba(255,122,47,0.36)" />
          <stop offset="100%" stopColor="rgba(255,122,47,0)" />
        </radialGradient>
        <linearGradient id="crack" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ffd9ac" />
          <stop offset="100%" stopColor="#ff7a2f" />
        </linearGradient>
        <filter id="soften" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="4.5" />
        </filter>
      </defs>

      <path
        d="M160 58C104 58 60 130 60 211c0 76 45 141 100 141s100-65 100-141C260 130 216 58 160 58Z"
        fill="url(#eggFill)"
        stroke="rgba(255,169,94,0.34)"
        strokeWidth="1.4"
      />
      <ellipse cx="160" cy="228" rx="86" ry="112" fill="url(#eggGlow)" />

      <g stroke="rgba(255,169,94,0.16)" strokeWidth="0.9" fill="none">
        {EDGES.filter(([, , lit]) => !lit).map(([a, b], i) => (
          <line
            key={`dim-${i}`}
            x1={NODES[a].x}
            y1={NODES[a].y}
            x2={NODES[b].x}
            y2={NODES[b].y}
          />
        ))}
      </g>

      <g fill="none" strokeLinecap="round">
        {EDGES.filter(([, , lit]) => lit).map(([a, b], i) => (
          <line
            key={`lit-${i}`}
            className={styles.synapse}
            style={{ animationDelay: `${i * 0.62}s` }}
            x1={NODES[a].x}
            y1={NODES[a].y}
            x2={NODES[b].x}
            y2={NODES[b].y}
            stroke="var(--ember)"
            strokeWidth="1.7"
          />
        ))}
      </g>

      {NODES.map((n, i) => (
        <g key={`node-${i}`}>
          <circle
            cx={n.x}
            cy={n.y}
            r={n.r * 2.6}
            fill="rgba(255,122,47,0.16)"
            filter="url(#soften)"
          />
          <circle
            className={styles.neuron}
            style={{ animationDelay: `${(i % 5) * 0.9}s` }}
            cx={n.x}
            cy={n.y}
            r={n.r}
            fill="var(--ember-pale)"
          />
        </g>
      ))}

      <path
        className={styles.crack}
        d="M214 92l-9 15 12 6-14 13 16 9-11 14"
        fill="none"
        stroke="url(#crack)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
