"use client";

import { useEffect, useRef } from "react";

import styles from "./NeuronGrid.module.css";

/**
 * One layer's neuron activations, drawn as a square of cells. Dark means the
 * neuron did not fire for this byte; the brighter the ember, the larger the
 * positive activation.
 */

const CELL = 9;
const GAP = 2;

function ember(t: number): string {
  // 0 -> deep coal, 0.5 -> ember orange, 1 -> pale hot
  const clamped = Math.max(0, Math.min(1, t));
  const r = 108 + 147 * clamped;
  const g = 44 + 173 * clamped ** 1.15;
  const b = 20 + 152 * clamped ** 2.2;
  return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
}

export function NeuronGrid({
  values,
  label,
  positive,
  total,
}: {
  values: Float32Array | null;
  label: string;
  positive: number;
  total: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cols = Math.ceil(Math.sqrt(total));
  const rows = Math.ceil(total / cols);
  const width = cols * (CELL + GAP) - GAP;
  const height = rows * (CELL + GAP) - GAP;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    let max = 0;
    if (values) {
      for (let i = 0; i < total; i++) if (values[i] > max) max = values[i];
    }

    for (let i = 0; i < total; i++) {
      const value = values ? values[i] : 0;
      const x = (i % cols) * (CELL + GAP);
      const y = Math.floor(i / cols) * (CELL + GAP);
      if (value > 0 && max > 0) {
        // sqrt keeps small-but-firing neurons visible
        context.fillStyle = ember(Math.sqrt(value / max));
      } else {
        context.fillStyle = "rgba(244, 232, 216, 0.045)";
      }
      context.fillRect(x, y, CELL, CELL);
    }
  }, [values, cols, total, width, height]);

  const percent = total > 0 ? (positive / total) * 100 : 0;

  return (
    <div className={styles.wrap}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        style={{ aspectRatio: `${width} / ${height}` }}
        role="img"
        aria-label={`${label}: ${positive} of ${total} neurons positive`}
      />
      <div className={styles.meta}>
        <span className={styles.label}>{label}</span>
        <span className={styles.value}>
          {percent.toFixed(1)}
          <span className={styles.unit}>%</span>
        </span>
      </div>
    </div>
  );
}
