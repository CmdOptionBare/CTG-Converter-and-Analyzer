import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { CorrSurface } from "@/lib/ctg-analysis/windows";
import { clockOf, formatDe, formatSigned } from "@/lib/ctg-analysis/format";
import { mix, palette, setupCanvas, useElementWidth } from "./canvas";
import { HoverTip } from "./ChartFrame";

const HEIGHT = 320;
const PAD = { l: 64, r: 16, t: 12, b: 32 };

export function CrossCorrHeatmap({
  surface,
  startedAt,
  activeT0,
}: {
  surface: CorrSurface;
  startedAt: string;
  activeT0?: number | null;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrapRef);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width < 80) return;
    const ctx = setupCanvas(canvas, width, HEIGHT);
    if (!ctx) return;
    drawHeat(ctx, width, surface, startedAt, activeT0 ?? null);
    canvas.dataset.ready = surface.t0.length ? "1" : "0";
  }, [width, surface, startedAt, activeT0]);

  function pointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const cell = cellAt(surface, width, x, y);
    if (!cell) {
      setTip(null);
      return;
    }
    const value = surface.r[cell.row]?.[cell.col];
    setTip({
      x,
      y,
      text: `${clockOf(startedAt, surface.t0[cell.row]).slice(0, 5)}\nLag ${formatSigned(surface.lags[cell.col], 0)} s\nr ${formatDe(value, 2)}`,
    });
  }

  return (
    <div ref={wrapRef} className="relative w-full">
      {surface.t0.length ? (
        <canvas
          ref={canvasRef}
          className="block w-full"
          style={{ height: HEIGHT }}
          onPointerMove={pointer}
          onPointerLeave={() => setTip(null)}
          role="img"
          aria-label="Kreuzkorrelation von Zeit und Lag"
        />
      ) : (
        <p className="text-sm text-paper/60">Die Aufnahme ist kürzer als ein 10-Minuten-Fenster.</p>
      )}
      {tip ? <HoverTip x={tip.x} y={tip.y} text={tip.text} /> : null}
    </div>
  );
}

function cellAt(surface: CorrSurface, width: number, x: number, y: number): { row: number; col: number } | null {
  const rows = surface.t0.length;
  const cols = surface.lags.length;
  if (!rows || !cols) return null;
  const plotW = width - PAD.l - PAD.r;
  const plotH = HEIGHT - PAD.t - PAD.b;
  const col = Math.floor(((x - PAD.l) / plotW) * cols);
  const row = Math.floor(((y - PAD.t) / plotH) * rows);
  if (row < 0 || col < 0 || row >= rows || col >= cols) return null;
  return { row, col };
}

function drawHeat(
  ctx: CanvasRenderingContext2D,
  width: number,
  surface: CorrSurface,
  startedAt: string,
  activeT0: number | null,
) {
  ctx.clearRect(0, 0, width, HEIGHT);
  ctx.fillStyle = palette.bg;
  ctx.fillRect(0, 0, width, HEIGHT);
  const rows = surface.t0.length;
  const cols = surface.lags.length;
  if (!rows || !cols) return;
  const plotW = width - PAD.l - PAD.r;
  const plotH = HEIGHT - PAD.t - PAD.b;
  const cw = plotW / cols;
  const rh = plotH / rows;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const value = surface.r[row]?.[col];
      ctx.fillStyle = value == null ? palette.bg : value >= 0 ? mix(palette.raised, palette.grid, value) : mix(palette.raised, palette.copper, -value);
      ctx.fillRect(PAD.l + col * cw, PAD.t + row * rh, cw + 0.4, rh + 0.4);
    }
    const lag = surface.bestNegLag[row];
    const col = lag == null ? -1 : surface.lags.indexOf(lag);
    if (col >= 0) {
      ctx.strokeStyle = palette.paper;
      ctx.strokeRect(PAD.l + col * cw + 0.5, PAD.t + row * rh + 0.5, Math.max(1, cw - 1), Math.max(1, rh - 1));
    }
    if (activeT0 != null && Math.abs(surface.t0[row] - activeT0) < 1) {
      ctx.strokeStyle = palette.copper;
      ctx.strokeRect(PAD.l, PAD.t + row * rh, plotW, rh);
    }
  }

  ctx.fillStyle = "rgba(243,241,231,0.45)";
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const step = Math.max(1, Math.ceil(rows / 8));
  for (let row = 0; row < rows; row += step) {
    ctx.fillText(clockOf(startedAt, surface.t0[row]).slice(0, 5), PAD.l - 6, PAD.t + (row + 0.5) * rh);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (const lag of [-60, -30, 0, 30, 60, 90, 120]) {
    const col = surface.lags.indexOf(lag);
    if (col < 0) continue;
    ctx.fillText(String(lag), PAD.l + (col + 0.5) * cw, HEIGHT - 10);
  }
}
