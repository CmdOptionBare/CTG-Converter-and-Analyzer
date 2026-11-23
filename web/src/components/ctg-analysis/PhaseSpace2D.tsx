import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { CtgSample } from "@/lib/ctg/types";
import { isSample } from "@/lib/ctg-analysis/metrics";
import { clockOf, formatDe } from "@/lib/ctg-analysis/format";
import { palette, setupCanvas, useElementWidth } from "./canvas";
import { HoverTip } from "./ChartFrame";

type Mode = "lag" | "toco";

type Point = { x: number; y: number; t: number };

type Props = {
  fhr: CtgSample[];
  toco: CtgSample[];
  dt: number;
  originSec: number;
  startedAt: string;
};

const HEIGHT = 320;
const PAD = { l: 48, r: 16, t: 12, b: 36 };

export function PhaseSpace2D({ fhr, toco, dt, originSec, startedAt }: Props) {
  const [mode, setMode] = useState<Mode>("lag");
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const width = useElementWidth(wrapRef);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const points = useMemo(() => buildPoints(fhr, toco, dt, originSec, mode), [fhr, toco, dt, originSec, mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width < 40) return;
    const ctx = setupCanvas(canvas, width, HEIGHT);
    if (!ctx) return;
    drawPhase(ctx, width, points, mode);
    canvas.dataset.ready = "1";
  }, [width, points, mode]);

  function pointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || !points.length) return;
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const scales = scalesOf(points, width);
    let best = -1;
    let bestD = 18 ** 2;
    for (let i = 0; i < points.length; i++) {
      const px = scales.xOf(points[i].x);
      const py = scales.yOf(points[i].y);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best < 0) {
      setTip(null);
      return;
    }
    const point = points[best];
    const text =
      mode === "lag"
        ? `${clockOf(startedAt, point.t)}\nFHF ${formatDe(point.x, 0)} → ${formatDe(point.y, 0)}`
        : `${clockOf(startedAt, point.t)}\nToco ${formatDe(point.x, 1)}\nFHF ${formatDe(point.y, 0)}`;
    setTip({ x: mx, y: my, text });
  }

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-1" role="group" aria-label="Phasenraum">
        <ModeButton active={mode === "lag"} onClick={() => setMode("lag")}>
          FHF(t) gegen FHF(t+1)
        </ModeButton>
        <ModeButton active={mode === "toco"} onClick={() => setMode("toco")}>
          Toco gegen FHF
        </ModeButton>
      </div>
      <div ref={wrapRef} className="relative w-full">
        <canvas
          ref={canvasRef}
          className="block w-full"
          style={{ height: HEIGHT }}
          onPointerMove={pointer}
          onPointerLeave={() => setTip(null)}
          role="img"
          aria-label={mode === "lag" ? "Phasenraum der fetalen Herzfrequenz" : "Phasenraum Toco gegen FHF"}
        />
        {tip ? <HoverTip x={tip.x} y={tip.y} text={tip.text} /> : null}
        {!points.length ? <p className="mt-2 text-xs text-paper/50">Zu wenige gültige Paare in diesem Ausschnitt.</p> : null}
      </div>
    </div>
  );
}

function ModeButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        active
          ? "rounded-card border border-copper px-3 py-1.5 text-xs text-copper"
          : "rounded-card border border-line px-3 py-1.5 text-xs text-paper/60"
      }
    >
      {children}
    </button>
  );
}

function buildPoints(fhr: CtgSample[], toco: CtgSample[], dt: number, originSec: number, mode: Mode): Point[] {
  const out: Point[] = [];
  if (mode === "lag") {
    for (let i = 0; i < fhr.length - 1; i++) {
      const a = fhr[i];
      const b = fhr[i + 1];
      if (!isSample(a) || !isSample(b)) continue;
      out.push({ x: a, y: b, t: originSec + i * dt });
    }
    return out;
  }
  const n = Math.min(fhr.length, toco.length);
  for (let i = 0; i < n; i++) {
    const heart = fhr[i];
    const contraction = toco[i];
    if (!isSample(heart) || !isSample(contraction)) continue;
    out.push({ x: contraction, y: heart, t: originSec + i * dt });
  }
  return out;
}

function scalesOf(points: Point[], width: number) {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const point of points) {
    if (point.x < x0) x0 = point.x;
    if (point.x > x1) x1 = point.x;
    if (point.y < y0) y0 = point.y;
    if (point.y > y1) y1 = point.y;
  }
  if (!points.length) {
    x0 = 0;
    x1 = 1;
    y0 = 0;
    y1 = 1;
  }
  const xPad = Math.max(1, (x1 - x0) * 0.08);
  const yPad = Math.max(1, (y1 - y0) * 0.08);
  x0 -= xPad;
  x1 += xPad;
  y0 -= yPad;
  y1 += yPad;
  const plotW = Math.max(1, width - PAD.l - PAD.r);
  const plotH = HEIGHT - PAD.t - PAD.b;
  return {
    x0,
    x1,
    y0,
    y1,
    xOf: (value: number) => PAD.l + ((value - x0) / (x1 - x0)) * plotW,
    yOf: (value: number) => PAD.t + (1 - (value - y0) / (y1 - y0)) * plotH,
  };
}

function drawPhase(ctx: CanvasRenderingContext2D, width: number, points: Point[], mode: Mode) {
  ctx.clearRect(0, 0, width, HEIGHT);
  ctx.fillStyle = palette.bg;
  ctx.fillRect(0, 0, width, HEIGHT);
  const scales = scalesOf(points, width);
  ctx.strokeStyle = palette.line;
  ctx.strokeRect(PAD.l, PAD.t, width - PAD.l - PAD.r, HEIGHT - PAD.t - PAD.b);

  if (mode === "lag") {
    ctx.strokeStyle = "rgba(243,241,231,0.25)";
    ctx.beginPath();
    ctx.moveTo(scales.xOf(scales.x0), scales.yOf(scales.x0));
    ctx.lineTo(scales.xOf(scales.x1), scales.yOf(scales.x1));
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(243,241,231,0.45)";
  for (const point of points) {
    ctx.fillRect(scales.xOf(point.x), scales.yOf(point.y), 1.6, 1.6);
  }

  ctx.fillStyle = "rgba(243,241,231,0.55)";
  ctx.font = "11px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(mode === "lag" ? "FHF(t)  bpm" : "Toco  0–100", PAD.l + (width - PAD.l - PAD.r) / 2, HEIGHT - 10);
  ctx.save();
  ctx.translate(16, PAD.t + (HEIGHT - PAD.t - PAD.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(mode === "lag" ? "FHF(t+1)  bpm" : "FHF  bpm", 0, 0);
  ctx.restore();

  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(243,241,231,0.4)";
  ctx.fillText(formatDe(scales.y1, mode === "toco" ? 0 : 0), PAD.l - 6, scales.yOf(scales.y1));
  ctx.fillText(formatDe(scales.y0, 0), PAD.l - 6, scales.yOf(scales.y0));
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(formatDe(scales.x0, mode === "toco" ? 0 : 0), PAD.l, HEIGHT - 22);
  ctx.textAlign = "right";
  ctx.fillText(formatDe(scales.x1, mode === "toco" ? 0 : 0), width - PAD.r, HEIGHT - 22);
}
