"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { paramCount, TOY_CONFIG, type ToyParams } from "@/lib/bdh/model";
import {
  decodeBytesForDisplay,
  encodeBytes,
  kvCacheFloats,
  rhoFloats,
  ToySession,
  type StepTrace,
} from "@/lib/bdh/recurrent";
import { loadToyWeights, type ToyWeightsFile } from "@/lib/bdh/weights";
import { formatCount, TOTAL_PARAMS } from "@/lib/facts";
import { NeuronGrid } from "./NeuronGrid";
import styles from "./ToyLab.module.css";

type View = "y" | "x" | "gate";
type Phase = "idle" | "reading" | "writing" | "done";

const VIEWS: { id: View; label: string; blurb: string }[] = [
  {
    id: "x",
    label: "x — fired",
    blurb: "ReLU(x @ Dx): which neurons this byte woke up",
  },
  {
    id: "gate",
    label: "gate — retrieved",
    blurb: "ReLU(LN(rho^T q) @ Dy): what working memory asked for",
  },
  {
    id: "y",
    label: "y — both",
    blurb: "gate * x: the sparse positive vector the layer passes on",
  },
];

interface RunState {
  session: ToySession;
  bytes: Uint8Array;
  read: number;
  produced: number[];
  logits: Float32Array | null;
  options: { maxNew: number; temperature: number; topK: number };
}

type Fractions = Record<View, number>;

function signed(delta: number): string {
  if (Math.abs(delta) < 5e-5) return "no change";
  return `${delta > 0 ? "+" : "\u2212"}${Math.abs(delta).toFixed(4)}`;
}

/** Share of positive entries across all layers, for each of the three vectors. */
function positiveFractions(trace: StepTrace): Fractions {
  let x = 0;
  let gate = 0;
  let y = 0;
  for (const layer of trace.layers) {
    x += layer.xPositive;
    gate += layer.gatePositive;
    y += layer.yPositive;
  }
  const total = trace.layers.length * TOY_CONFIG.n;
  return { x: x / total, gate: gate / total, y: y / total };
}

export function ToyLab() {
  const [weights, setWeights] = useState<{
    file: ToyWeightsFile;
    params: ToyParams;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState("To be or ");
  const [maxNew, setMaxNew] = useState(96);
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState(3);
  const [seed, setSeed] = useState(1337);
  const [view, setView] = useState<View>("y");

  const [phase, setPhase] = useState<Phase>("idle");
  const [outBytes, setOutBytes] = useState<number[]>([]);
  const [trace, setTrace] = useState<StepTrace | null>(null);
  const [tokensSeen, setTokensSeen] = useState(0);
  const [activity, setActivity] = useState<Fractions[]>([]);

  const runRef = useRef<RunState | null>(null);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    loadToyWeights(controller.signal)
      .then(setWeights)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => controller.abort();
  }, []);

  const stop = useCallback(() => {
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  }, []);

  useEffect(() => stop, [stop]);

  const tick = useCallback(() => {
    const run = runRef.current;
    if (!run) return;

    const record = (next: StepTrace) => {
      run.logits = next.logits;
      setTrace(next);
      setTokensSeen(run.session.position);
      setActivity((previous) => [...previous, positiveFractions(next)]);
    };

    if (run.read < run.bytes.length) {
      record(run.session.feed(run.bytes[run.read], true));
      run.read += 1;
      if (run.read >= run.bytes.length) setPhase("writing");
      frameRef.current = requestAnimationFrame(tick);
      return;
    }

    if (run.produced.length >= run.options.maxNew || !run.logits) {
      setPhase("done");
      frameRef.current = null;
      return;
    }

    const next = run.session.sample(run.logits, {
      temperature: run.options.temperature,
      topK: run.options.topK,
    });
    run.produced.push(next);
    setOutBytes([...run.produced]);
    record(run.session.feed(next, true));
    frameRef.current = requestAnimationFrame(tick);
  }, []);

  const start = useCallback(() => {
    if (!weights) return;
    const bytes = encodeBytes(prompt);
    if (bytes.length === 0) return;
    stop();
    runRef.current = {
      session: new ToySession(weights.params, TOY_CONFIG, seed),
      bytes,
      read: 0,
      produced: [],
      logits: null,
      options: { maxNew, temperature, topK },
    };
    setOutBytes([]);
    setActivity([]);
    setTrace(null);
    setTokensSeen(0);
    setPhase("reading");
    frameRef.current = requestAnimationFrame(tick);
  }, [weights, prompt, seed, maxNew, temperature, topK, stop, tick]);

  const reset = useCallback(() => {
    stop();
    runRef.current = null;
    setPhase("idle");
    setOutBytes([]);
    setActivity([]);
    setTrace(null);
    setTokensSeen(0);
  }, [stop]);

  const promptBytes = useMemo(() => Array.from(encodeBytes(prompt)), [prompt]);
  const busy = phase === "reading" || phase === "writing";
  const toyParams = paramCount(TOY_CONFIG);
  const stateFloats = rhoFloats(TOY_CONFIG);
  const cacheFloats = kvCacheFloats(TOY_CONFIG, tokensSeen);
  const crossover = TOY_CONFIG.n / 2;
  const activeView = VIEWS.find((v) => v.id === view)!;

  if (error) {
    return (
      <div className="card">
        <p className="label label-ember">Could not start</p>
        <p className="dim" style={{ marginBottom: 0 }}>
          The trained weights failed to load: {error}. They live at{" "}
          <code>web/public/bdh-toy-weights.json</code> and are regenerated with{" "}
          <code>npm run toy:train</code>.
        </p>
      </div>
    );
  }

  return (
    <div className={styles.lab}>
      <div className={`card card-ember ${styles.controls}`}>
        <div className={styles.promptRow}>
          <label className={styles.field}>
            <span className="label">Prompt (bytes, not tokens)</span>
            <input
              className={styles.input}
              value={prompt}
              spellCheck={false}
              maxLength={120}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="To be or "
            />
          </label>
          <div className={styles.buttons}>
            <button
              className="button"
              onClick={start}
              disabled={!weights || promptBytes.length === 0}
            >
              {busy ? "Running…" : weights ? "Generate" : "Loading weights…"}
            </button>
            <button
              className="button button-ghost"
              onClick={reset}
              disabled={phase === "idle"}
            >
              Reset
            </button>
          </div>
        </div>

        <div className={styles.sliders}>
          <Slider
            label="new bytes"
            value={maxNew}
            min={16}
            max={120}
            step={8}
            onChange={setMaxNew}
          />
          <Slider
            label="temperature"
            value={temperature}
            min={0.2}
            max={1.6}
            step={0.1}
            format={(v) => v.toFixed(1)}
            onChange={setTemperature}
          />
          <Slider
            label="top-k"
            value={topK}
            min={1}
            max={12}
            step={1}
            onChange={setTopK}
          />
          <Slider
            label="seed"
            value={seed}
            min={1}
            max={9999}
            step={1}
            onChange={setSeed}
          />
        </div>

        <div className={styles.byteStrip}>
          <span className="label" style={{ flex: "none" }}>
            {promptBytes.length} bytes
          </span>
          <div className={styles.bytes}>
            {promptBytes.map((byte, index) => (
              <span
                key={`${index}-${byte}`}
                className={[
                  styles.byte,
                  phase === "reading" && trace?.position === index
                    ? styles.byteActive
                    : "",
                  (phase === "writing" || phase === "done") ? styles.byteRead : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
              >
                <span className={styles.byteChar}>
                  {byte === 32 ? "·" : decodeBytesForDisplay([byte])}
                </span>
                <span className={styles.byteHex}>
                  {byte.toString(16).padStart(2, "0")}
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className={`card ${styles.output}`}>
        <div className={styles.outputHead}>
          <p className="label label-ember">
            {phase === "idle"
              ? "Output"
              : phase === "reading"
                ? "Reading your prompt, byte by byte"
                : phase === "writing"
                  ? "Generating"
                  : "Output"}
          </p>
          <span className="pill">
            {tokensSeen} bytes through the state
          </span>
        </div>
        <pre className={`sample ${styles.sampleBox}`}>
          <span className={styles.promptText}>{prompt}</span>
          {decodeBytesForDisplay(outBytes)}
          {busy ? <span className={styles.caret} /> : null}
          {phase === "idle" ? (
            <span className="dim">
              {"\n"}Press generate. The model reads your prompt one byte at a
              time, and every byte updates rho.
            </span>
          ) : null}
        </pre>
      </div>

      <div className={`card ${styles.neurons}`}>
        <div className={styles.neuronsHead}>
          <div>
            <p className="label label-ember">
              Positive activations, layer by layer
            </p>
            <p className={styles.neuronsBlurb}>{activeView.blurb}</p>
          </div>
          <div className={styles.toggle} role="group" aria-label="Which vector to show">
            {VIEWS.map((option) => (
              <button
                key={option.id}
                className={[
                  styles.toggleButton,
                  view === option.id ? styles.toggleActive : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                onClick={() => setView(option.id)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.grids}>
          {Array.from({ length: TOY_CONFIG.layers }, (_, layer) => {
            const data = trace?.layers[layer];
            const values = data
              ? view === "x"
                ? data.x
                : view === "gate"
                  ? data.gate
                  : data.y
              : null;
            const positive = data
              ? view === "x"
                ? data.xPositive
                : view === "gate"
                  ? data.gatePositive
                  : data.yPositive
              : 0;
            return (
              <NeuronGrid
                key={layer}
                label={`layer ${layer + 1}`}
                values={values}
                positive={positive}
                total={TOY_CONFIG.n}
              />
            );
          })}
        </div>

        <ActivityStrip series={activity} view={view} />
      </div>

      <div className={`grid grid-2 ${styles.bottom}`}>
        <div className="card">
          <p className="label label-ember">Working memory vs a KV cache</p>
          <div className={styles.meter}>
            <MeterRow
              name="rho (this model)"
              value={stateFloats}
              max={Math.max(stateFloats, cacheFloats, 1)}
              tone="ember"
              note="fixed: layers x n x d"
            />
            <MeterRow
              name={`KV cache at ${tokensSeen} bytes`}
              value={cacheFloats}
              max={Math.max(stateFloats, cacheFloats, 1)}
              tone="cool"
              note="grows: 2 x layers x T x d"
            />
          </div>
          <p className="footnote" style={{ marginBottom: 0 }}>
            The two are equal at T = n / 2, which is {crossover} bytes at this
            toy&apos;s n = {TOY_CONFIG.n} — and {formatCount(32768 / 2)} bytes at
            the 25M model&apos;s n = {formatCount(32768)}. The crossover does not
            depend on d or on depth. Past it, rho stops growing and the cache does
            not.
          </p>
        </div>

        <div className="card">
          <p className="label label-ember">What you are actually running</p>
          {weights ? (
            <>
              <table className={`table ${styles.factTable}`}>
                <tbody>
                  <tr>
                    <td className="dim">shape</td>
                    <td className={styles.factValue}>
                      n = {TOY_CONFIG.n}, d = {TOY_CONFIG.d},{" "}
                      {TOY_CONFIG.layers} layers, 1 head
                    </td>
                  </tr>
                  <tr>
                    <td className="dim">parameters</td>
                    <td className={styles.factValue}>
                      {formatCount(toyParams)}
                      <span className="dim">
                        {" "}
                        — {Math.round(TOTAL_PARAMS / toyParams)}x smaller than the
                        repo model
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="dim">trained</td>
                    <td className={styles.factValue}>
                      {formatCount(weights.file.training.steps)} steps, block{" "}
                      {weights.file.training.block}, batch{" "}
                      {weights.file.training.batch}
                    </td>
                  </tr>
                  <tr>
                    <td className="dim">loss</td>
                    <td className={styles.factValue}>
                      {weights.file.training.firstLoss.toFixed(2)} →{" "}
                      {weights.file.training.trainLoss.toFixed(2)} train,{" "}
                      {weights.file.training.valLoss.toFixed(2)} val
                    </td>
                  </tr>
                  <tr>
                    <td className="dim">int8 packing</td>
                    <td className={styles.factValue}>
                      val {weights.file.training.valLossInt8.toFixed(4)}{" "}
                      <span className="dim">
                        ({signed(
                          weights.file.training.valLossInt8 -
                            weights.file.training.valLoss,
                        )}{" "}
                        on the same windows)
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="dim">y positive</td>
                    <td className={styles.factValue}>
                      {(
                        weights.file.training.positiveActivations.y * 100
                      ).toFixed(1)}
                      % measured on held-out text
                    </td>
                  </tr>
                </tbody>
              </table>
              <p className="footnote" style={{ marginBottom: 0 }}>
                Trained by <code>web/scripts/train-toy.ts</code> on byte-level
                tiny Shakespeare — the same corpus <code>train.py</code> uses,
                and the same AdamW settings. It is far below the scale at which
                the paper measured about 5% sparsity, so do not read this
                model&apos;s percentages as that result. Byte output can be
                invalid UTF-8; non-printing bytes are escaped the way{" "}
                <code>train.py</code> prints them.
              </p>
            </>
          ) : (
            <p className="dim">Loading…</p>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * How the firing rate moves from byte to byte. The absolute numbers barely
 * change, so the bars are scaled to the range actually observed in this run —
 * the range itself is printed, so nothing is hidden by the zoom.
 */
function ActivityStrip({ series, view }: { series: Fractions[]; view: View }) {
  const shown = series.map((f) => f[view]).slice(-160);
  // The very first byte reads an empty rho, so its gate — and therefore y — is
  // exactly zero. Scaling to that one degenerate value would flatten the rest.
  const scored = shown.filter((value) => value > 0);
  const low = scored.length > 0 ? Math.min(...scored) : 0;
  const high = scored.length > 0 ? Math.max(...scored) : 0;
  const spread = Math.max(high - low, 1e-6);

  return (
    <div className={styles.activity}>
      <div className={styles.activityLabel}>
        <span className="label">{view} positive, per byte</span>
        <span className={styles.activityRange}>
          {shown.length > 0
            ? `${(low * 100).toFixed(1)}–${(high * 100).toFixed(1)}%`
            : "—"}
        </span>
      </div>
      <div className={styles.activityBars}>
        {shown.map((value, index) => (
          <span
            key={index}
            className={styles.activityBar}
            style={{
              height: `${
                12 +
                Math.max(0, Math.min(1, (value - low) / spread)) * 88
              }%`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <label className={styles.slider}>
      <span className={styles.sliderHead}>
        <span className="label">{label}</span>
        <span className={styles.sliderValue}>
          {format ? format(value) : value}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className={styles.range}
      />
    </label>
  );
}

function MeterRow({
  name,
  value,
  max,
  tone,
  note,
}: {
  name: string;
  value: number;
  max: number;
  tone: "ember" | "cool";
  note: string;
}) {
  return (
    <div className={styles.meterRow}>
      <div className={styles.meterHead}>
        <span className={styles.meterName}>{name}</span>
        <span className={styles.meterValue}>
          {formatCount(value)} <span className="dim">floats</span>
        </span>
      </div>
      <div className={styles.meterTrack}>
        <div
          className={tone === "ember" ? styles.meterFillEmber : styles.meterFillCool}
          style={{ width: `${Math.min(100, (value / max) * 100)}%` }}
        />
      </div>
      <span className={styles.meterNote}>{note}</span>
    </div>
  );
}
