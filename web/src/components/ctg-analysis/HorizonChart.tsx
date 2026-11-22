import { useEffect, useRef, useState } from "react";
import type { CtgSample } from "@/lib/ctg/types";
import { isSample } from "@/lib/ctg-analysis/metrics";
import { clockOf, formatDe, formatSigned } from "@/lib/ctg-analysis/format";
import { palette, setupCanvas, useElementWidth } from "./canvas";
import { HoverTip } from "./ChartFrame";

const HEIGHT = 210;
const PAD = { l: 46, r: 12, t: 14, b: 28 };

type Props = {
  fhr: CtgSample[];
  dt: number;
  durationSec: number;
  baseline: number | null;
  startedAt: string;
  highlight?: { t0: number; t1: number } | null;
  cursorSec?: number | null;
  onCursor?: (t: number | null) => void;
};

export function HorizonChart({ fhr, dt, durationSec, baseline, startedAt, highlight, cursorSec, onCursor }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrapRef);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width < 40) return;
    let cancelled = false;
    const paint = () => {
      if (cancelled) return;
      const ctx = setupCanvas(canvas, width, HEIGHT);
      if (!ctx) return;
      drawHorizon(ctx, width, { fhr, dt, durationSec, baseline, startedAt, highlight, cursorSec });
      canvas.dataset.ready = "1";
    };
    paint();
    void document.fonts?.ready.then(paint);
    return () => {
      cancelled = true;
    };
  }, [width, fhr, dt, durationSec, baseline, startedAt, highlight, cursorSec]);

  function pointer(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || baseline == null || durationSec <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const plotW = width - PAD.l - PAD.r;
    const t = ((x - PAD.l) / Math.max(1, plotW)) * durationSec;
    if (t < 0 || t > durationSec) {
      setTip(null);
      onCursor?.(null);
      return;
    }
    const index = Math.min(fhr.length - 1, Math.max(0, Math.round(t / dt)));
    const value = fhr[index];
    const delta = isSample(value) ? value - baseline : null;
    const text = isSample(value)
      ? `${clockOf(startedAt, index * dt)}\nFHF ${formatDe(value, 0)} bpm\nΔ ${formatSigned(delta, 0)}`
      : `${clockOf(startedAt, index * dt)}\nLücke`;
    setTip({ x, y: event.clientY - rect.top, text });
    onCursor?.(Math.round(t));
  }

  return (
    <div ref={wrapRef} className="relative w-full">
      <canvas
        ref={canvasRef}
        className="block w-full touch-none"
        style={{ height: HEIGHT }}
        onPointerMove={pointer}
        onPointerLeave={() => {
          setTip(null);
          onCursor?.(null);
        }}
        role="img"
        aria-label="Horizont der fetalen Herzfrequenz um die Basis"
      />
      {tip ? <HoverTip x={tip.x} y={tip.y} text={tip.text} /> : null}
    </div>
  );
}

function drawHorizon(
  ctx: CanvasRenderingContext2D,
  width: number,
  props: Omit<Props, "onCursor">,
) {
  const { fhr, dt, durationSec, baseline, startedAt, highlight, cursorSec } = props;
  ctx.clearRect(0, 0, width, HEIGHT);
  ctx.fillStyle = palette.bg;
  ctx.fillRect(0, 0, width, HEIGHT);
  const plotW = Math.max(1, width - PAD.l - PAD.r);
  const plotH = HEIGHT - PAD.t - PAD.b;
  const midY = PAD.t + plotH / 2;

  if (baseline == null || !fhr.length || !(durationSec > 0)) {
    ctx.fillStyle = "rgba(243,241,231,0.5)";
    ctx.font = "13px 'IBM Plex Sans', sans-serif";
    ctx.fillText("Keine gültige FHF.", PAD.l, midY);
    return;
  }

  const yMax = chooseYMax(fhr, baseline);
  const yScale = plotH / 2 / yMax;
  const bins = Math.max(1, Math.floor(plotW));
  const min = new Array<number>(bins).fill(Infinity);
  const max = new Array<number>(bins).fill(-Infinity);
  const sum = new Array<number>(bins).fill(0);
  const count = new Array<number>(bins).fill(0);
  for (let i = 0; i < fhr.length; i++) {
    const value = fhr[i];
    if (!isSample(value)) continue;
    const t = i * dt;
    const x = Math.min(bins - 1, Math.max(0, Math.floor((t / durationSec) * bins)));
    const dev = value - baseline;
    if (dev < min[x]) min[x] = dev;
    if (dev > max[x]) max[x] = dev;
    sum[x] += dev;
    count[x] += 1;
  }

  if (highlight && (highlight.t0 > 0.5 || highlight.t1 < durationSec - 0.5)) {
    const x0 = PAD.l + (highlight.t0 / durationSec) * plotW;
    const x1 = PAD.l + (highlight.t1 / durationSec) * plotW;
    ctx.fillStyle = "rgba(243,241,231,0.05)";
    ctx.fillRect(x0, PAD.t, Math.max(1, x1 - x0), plotH);
    ctx.strokeStyle = "rgba(196,98,45,0.85)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, PAD.t + 0.5, Math.max(1, x1 - x0), plotH);
  }

  ctx.strokeStyle = palette.line;
  ctx.lineWidth = 1;
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const ticks: number[] = [];
  for (let tick = -yMax; tick <= yMax + 1e-6; tick += 15) ticks.push(tick);
  for (const tick of ticks) {
    const y = midY - tick * yScale;
    ctx.beginPath();
    ctx.moveTo(PAD.l, y);
    ctx.lineTo(width - PAD.r, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(243,241,231,0.45)";
    ctx.fillText(tick > 0 ? `+${tick}` : String(tick), PAD.l - 6, y);
  }

  for (let x = 0; x < bins; x++) {
    if (!count[x]) continue;
    const px = PAD.l + (x / bins) * plotW;
    const colW = plotW / bins;
    if (max[x] > 0) paintBand(ctx, px, max[x], 1, yMax, midY, yScale, colW);
    if (min[x] < 0) paintBand(ctx, px, min[x], -1, yMax, midY, yScale, colW);
  }

  ctx.strokeStyle = "rgba(243,241,231,0.9)";
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  let pen = false;
  for (let x = 0; x < bins; x++) {
    const px = PAD.l + ((x + 0.5) / bins) * plotW;
    if (!count[x]) {
      pen = false;
      continue;
    }
    const y = midY - (sum[x] / count[x]) * yScale;
    if (!pen) ctx.moveTo(px, y);
    else ctx.lineTo(px, y);
    pen = true;
  }
  ctx.stroke();

  if (cursorSec != null) {
    const x = PAD.l + (cursorSec / durationSec) * plotW;
    ctx.strokeStyle = "rgba(243,241,231,0.75)";
    ctx.beginPath();
    ctx.moveTo(x, PAD.t);
    ctx.lineTo(x, PAD.t + plotH);
    ctx.stroke();
  }

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "rgba(243,241,231,0.45)";
  const xTicks = 6;
  for (let i = 0; i <= xTicks; i++) {
    const t = (durationSec * i) / xTicks;
    const x = PAD.l + (i / xTicks) * plotW;
    ctx.fillText(clockOf(startedAt, t).slice(0, 5), x, HEIGHT - 8);
  }
}

function chooseYMax(fhr: CtgSample[], baseline: number): number {
  let peak = 0;
  for (const value of fhr) {
    if (isSample(value)) peak = Math.max(peak, Math.abs(value - baseline));
  }
  return Math.min(60, Math.max(30, Math.ceil((peak + 0.1) / 15) * 15));
}

function paintBand(
  ctx: CanvasRenderingContext2D,
  x: number,
  to: number,
  sign: 1 | -1,
  yMax: number,
  midY: number,
  yScale: number,
  colW: number,
) {
  const mag = Math.min(Math.abs(to), yMax);
  const edges = [0, 15, 30, 45, 60, 75];
  for (let band = 0; band < edges.length - 1; band++) {
    if (mag <= edges[band]) break;
    const a = edges[band];
    const b = Math.min(mag, edges[band + 1]);
    const y0 = midY - sign * a * yScale;
    const y1 = midY - sign * b * yScale;
    ctx.fillStyle = bandColor(sign > 0, band);
    ctx.fillRect(x, Math.min(y0, y1), colW + 0.5, Math.max(1, Math.abs(y1 - y0)));
  }
}

function bandColor(positive: boolean, band: number): string {
  const alpha = [0.38, 0.66, 0.92][Math.min(2, band)];
  return positive ? `rgba(109, 138, 98, ${alpha})` : `rgba(196, 98, 45, ${alpha})`;
}
