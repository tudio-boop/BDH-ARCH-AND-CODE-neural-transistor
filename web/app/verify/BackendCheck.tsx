"use client";

import { useCallback, useEffect, useState } from "react";

import { GPU_TRUST_THRESHOLD } from "@/lib/bdh/backend";
import { TOY_CONFIG } from "@/lib/bdh/model";
import { compareToCpu, type BackendComparison } from "@/lib/bdh/selfcheck";
import { loadToyWeights } from "@/lib/bdh/weights";
import {
  requestGpuContext,
  WebGpuSession,
  WebGpuUnavailable,
} from "@/lib/bdh/webgpu/backend";
import styles from "./BackendCheck.module.css";

/** Longer and more varied than the lab's startup check: more positions, more chances to drift. */
const PROMPT =
  "To be or not to be, that is the question.\n" +
  "Whether 'tis nobler in the mind to suffer\n" +
  "The slings and arrows of outrageous fortune,\n" +
  "Or to take arms against a sea of troubles.\n";

type State =
  | { status: "loading" }
  | { status: "unavailable"; reason: string }
  | { status: "done"; adapter: string | null; result: BackendComparison }
  | { status: "error"; message: string };

export function BackendCheck() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let session: WebGpuSession | null = null;

    void (async () => {
      try {
        const { params } = await loadToyWeights();
        const context = await requestGpuContext();
        session = await WebGpuSession.create(params, TOY_CONFIG, context);
        const result = await compareToCpu(session, params, TOY_CONFIG, PROMPT);
        if (cancelled) return;
        setState({ status: "done", adapter: context.adapterLabel, result });
      } catch (cause) {
        if (cancelled) return;
        if (cause instanceof WebGpuUnavailable) {
          setState({ status: "unavailable", reason: cause.message });
        } else {
          setState({
            status: "error",
            message: cause instanceof Error ? cause.message : String(cause),
          });
        }
      } finally {
        session?.dispose();
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [nonce]);

  const rerun = useCallback(() => {
    setState({ status: "loading" });
    setNonce((n) => n + 1);
  }, []);

  if (state.status === "loading") {
    return (
      <div className="card">
        <p className="label label-ember">Running</p>
        <p className="dim" style={{ marginBottom: 0 }}>
          Feeding the same bytes through both backends…
        </p>
      </div>
    );
  }

  if (state.status === "unavailable") {
    return (
      <div className="card">
        <p className="label">Result</p>
        <h3 className={styles.verdict}>No WebGPU in this browser</h3>
        <p className="dim">
          {state.reason}. Nothing is broken: the lab runs the CPU TypeScript path
          instead, which is the implementation checked against{" "}
          <code>bdh.py</code>. Try a Chromium-based browser to exercise the GPU
          path.
        </p>
        <button className="button button-ghost" onClick={rerun}>
          Run again
        </button>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="card">
        <p className="label">Result</p>
        <h3 className={styles.verdict}>The check could not finish</h3>
        <p className="dim">{state.message}</p>
        <button className="button button-ghost" onClick={rerun}>
          Run again
        </button>
      </div>
    );
  }

  const { result, adapter } = state;
  const passed = result.relativeToScale < GPU_TRUST_THRESHOLD;

  return (
    <div className={`card ${passed ? "card-ember" : ""}`}>
      <div className={styles.head}>
        <div>
          <p className="label label-ember">Result</p>
          <h3 className={styles.verdict}>
            {passed
              ? "WebGPU matches the CPU reference"
              : "WebGPU disagrees with the CPU reference"}
          </h3>
        </div>
        <span className="pill">{passed ? "GPU trusted" : "GPU rejected"}</span>
      </div>

      <table className={`table ${styles.table}`}>
        <tbody>
          <tr>
            <td className="dim">adapter</td>
            <td className={styles.value}>
              {adapter ?? "(reported no description)"}
            </td>
          </tr>
          <tr>
            <td className="dim">bytes compared</td>
            <td className={styles.value}>{result.tokens}, teacher-forced</td>
          </tr>
          <tr>
            <td className="dim">largest |logit|</td>
            <td className={styles.value}>{result.logitScale.toExponential(4)}</td>
          </tr>
          <tr>
            <td className="dim">max difference</td>
            <td className={styles.value}>{result.maxAbsolute.toExponential(4)}</td>
          </tr>
          <tr>
            <td className="dim">relative to scale</td>
            <td className={styles.value}>
              {result.relativeToScale.toExponential(4)}{" "}
              <span className="dim">
                (threshold {GPU_TRUST_THRESHOLD.toExponential(0)})
              </span>
            </td>
          </tr>
          <tr>
            <td className="dim">mean difference</td>
            <td className={styles.value}>{result.meanAbsolute.toExponential(4)}</td>
          </tr>
          <tr>
            <td className="dim">max difference in y</td>
            <td className={styles.value}>
              {result.maxActivationDifference.toExponential(4)}
            </td>
          </tr>
          <tr>
            <td className="dim">same top-1 byte</td>
            <td className={styles.value}>
              {result.tokens - result.argmaxDisagreements}/{result.tokens} positions
            </td>
          </tr>
        </tbody>
      </table>

      <button className="button button-ghost" onClick={rerun}>
        Run again
      </button>
    </div>
  );
}
