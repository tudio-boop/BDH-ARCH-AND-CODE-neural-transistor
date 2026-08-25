"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  createBackend,
  createCpuBackend,
  type BdhBackend,
} from "@/lib/bdh/backend";
import { mulberry32 } from "@/lib/bdh/linalg";
import { paramCount, TOY_CONFIG, type ToyParams } from "@/lib/bdh/model";
import {
  decodeBytesForDisplay,
  encodeBytes,
  kvCacheFloats,
  rhoFloats,
  sampleFromLogits,
  type StepTrace,
} from "@/lib/bdh/recurrent";
import type { BackendComparison } from "@/lib/bdh/selfcheck";
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

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
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

  const [backendLabel, setBackendLabel] = useState<string | null>(null);
  const [backendDetail, setBackendDetail] = useState<string | null>(null);
  const [comparison, setComparison] = useState<BackendComparison | null>(null);

  const backendRef = useRef<BdhBackend | null>(null);
  const runIdRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      runIdRef.current += 1;
      backendRef.current?.dispose();
      backendRef.current = null;
    };
  }, []);

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

  // Pick a backend once the weights are in: WebGPU when the browser has it and
  // it agrees with the CPU reference, CPU otherwise.
  useEffect(() => {
    if (!weights) return;
    let cancelled = false;
    void createBackend(weights.params, TOY_CONFIG).then((selection) => {
      if (cancelled || !mountedRef.current) {
        selection.backend.dispose();
        return;
      }
      backendRef.current = selection.backend;
      setBackendLabel(selection.backend.label);
      setBackendDetail(selection.backend.detail);
      setComparison(selection.comparison);
    });
    return () => {
      cancelled = true;
    };
  }, [weights]);

  const publish = useCallback((step: StepTrace, position: number) => {
    setTrace(step);
    setTokensSeen(position);
    setActivity((previous) => [...previous, positiveFractions(step)]);
  }, []);

  const run = useCallback(
    async (attempt = 0): Promise<void> => {
      const backend = backendRef.current;
      if (!backend || !weights) return;
      const bytes = encodeBytes(prompt);
      if (bytes.length === 0) return;

      const runId = ++runIdRef.current;
      const alive = () => runIdRef.current === runId && mountedRef.current;

      backend.reset();
      setOutBytes([]);
      setActivity([]);
      setTrace(null);
      setTokensSeen(0);
      setPhase("reading");

      const rand = mulberry32(seed);
      const options = { temperature, topK };
      let logits: Float32Array | null = null;

      const step = async (token: number): Promise<StepTrace> => {
        const result = await backend.step(token, true);
        publish(result, backend.position);
        return result;
      };

      try {
        for (const token of bytes) {
          if (!alive()) return;
          await nextFrame();
          if (!alive()) return;
          logits = (await step(token)).logits;
        }
        if (!alive()) return;
        setPhase("writing");

        const produced: number[] = [];
        while (produced.length < maxNew && logits) {
          if (!alive()) return;
          await nextFrame();
          if (!alive()) return;
          const next = sampleFromLogits(logits, options, rand);
          produced.push(next);
          setOutBytes([...produced]);
          logits = (await step(next)).logits;
        }
        if (alive()) setPhase("done");
      } catch (cause) {
        if (!alive()) return;
        const message = cause instanceof Error ? cause.message : String(cause);
        // A GPU that fails mid-run is not worth retrying: drop to the CPU path
        // and start the run again so rho is rebuilt from the first byte.
        if (backend.kind === "webgpu" && attempt === 0) {
          backend.dispose();
          backendRef.current = createCpuBackend(
            weights.params,
            TOY_CONFIG,
            `WebGPU failed while running: ${message}`,
          );
          setBackendLabel("CPU");
          setBackendDetail(`WebGPU failed while running: ${message}`);
          setComparison(null);
          await run(attempt + 1);
          return;
        }
        setPhase("idle");
        setError(message);
      }
    },
    [weights, prompt, seed, temperature, topK, maxNew, publish],
  );

  const start = useCallback(() => {
    void run(0);
  }, [run]);

  const reset = useCallback(() => {
    runIdRef.current += 1;
    backendRef.current?.reset();
    setPhase("idle");
    setOutBytes([]);
    setActivity([]);
    setTrace(null);
    setTokensSeen(0);
  }, []);

  const promptBytes = useMemo(() => Array.from(encodeBytes(prompt)), [prompt]);
  const busy = phase === "reading" || phase === "writing";
  const ready = backendLabel !== null;
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
          {error}. The weights live at{" "}
          <code>web/public/bdh-toy-weights.json</code> and are regenerated with{" "}
          <code>npm run toy:train</code>.
        </p>
      </div>
    );
  }

  return (
    <div className={styles.lab}>
      <div className={styles.statusBar}>
        <span
          className={[
            styles.status,
            backendLabel === "WebGPU" ? styles.statusGpu : "",
            backendLabel === "CPU" ? styles.statusCpu : "",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          <span className={styles.statusDot} />
          {backendLabel === null ? "selecting compute…" : backendLabel}
        </span>
        {backendDetail ? (
          <span className={styles.statusDetail}>{backendDetail}</span>
        ) : null}
        {comparison ? (
          <span className={styles.statusDetail}>
            matches CPU to {comparison.relativeToScale.toExponential(1)} relative
            over {comparison.tokens} bytes
          </span>
        ) : null}
        <Link href="/verify" className={styles.statusLink}>
          full check
        </Link>
      </div>

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
              disabled={!ready || promptBytes.length === 0}
            >
              {busy ? "Running…" : ready ? "Generate" : "Loading model…"}
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
                  phase === "writing" || phase === "done" ? styles.byteRead : "",
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
          <span className="pill">{tokensSeen} bytes through the state</span>
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
          <div
            className={styles.toggle}
            role="group"
            aria-label="Which vector to show"
          >
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
                    <td className="dim">compute</td>
                    <td className={styles.factValue}>
                      {backendLabel ?? "selecting…"}
                      {backendDetail ? (
                        <span className="dim"> — {backendDetail}</span>
                      ) : null}
                    </td>
                  </tr>
                  {comparison ? (
                    <tr>
                      <td className="dim">gpu vs cpu</td>
                      <td className={styles.factValue}>
                        max {comparison.maxAbsolute.toExponential(2)} on logits of
                        scale {comparison.logitScale.toFixed(1)} (
                        {comparison.relativeToScale.toExponential(1)} relative),
                        same top-1 byte at{" "}
                        {comparison.tokens - comparison.argmaxDisagreements}/
                        {comparison.tokens} positions
                      </td>
                    </tr>
                  ) : null}
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
                        (
                        {signed(
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
                12 + Math.max(0, Math.min(1, (value - low) / spread)) * 88
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
          className={
            tone === "ember" ? styles.meterFillEmber : styles.meterFillCool
          }
          style={{ width: `${Math.min(100, (value / max) * 100)}%` }}
        />
      </div>
      <span className={styles.meterNote}>{note}</span>
    </div>
  );
}
