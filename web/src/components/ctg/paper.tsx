import { useEffect, useRef } from "react";
import type { CtgSession } from "@/lib/ctg/types";
import { clockOf } from "@/lib/ctg/analyze";

type Props = {
  session: CtgSession;
  pxPerCm: number;
  showPhoto: boolean;
  photoOpacity: number;
  shiftSec: number;
  showFhr: boolean;
  showMhr: boolean;
  showToco: boolean;
  activeScan: string | null;
  onCursor: (t: number | null) => void;
  scrollToSec?: number;
};

const FHR_TOP = 210;
const FHR_BOT = 50;
const TOCO_TOP = 12;
const TOCO_BOT = 0;

export function CtgPaper({
  session,
  pxPerCm,
  showPhoto,
  photoOpacity,
  shiftSec,
  showFhr,
  showMhr,
  showToco,
  activeScan,
  onCursor,
  scrollToSec,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const images = useRef<Map<string, HTMLImageElement>>(new Map());

  const fhrCm = (FHR_TOP - FHR_BOT) / 20;
  const tocoCm = (TOCO_TOP - TOCO_BOT) / 2;
  const gapCm = 0.7;
  const padTop = 18;
  const padBot = 22;
  const width = Math.ceil((session.durationSec / 60) * pxPerCm + 24);
  const fhrY = padTop;
  const fhrH = fhrCm * pxPerCm;
  const tocoY = fhrY + fhrH + gapCm * pxPerCm;
  const tocoH = tocoCm * pxPerCm;
  const height = Math.ceil(tocoY + tocoH + padBot);

  const drawRef = useRef<() => void>(() => {});
  drawRef.current = draw;

  useEffect(() => {
    let alive = true;
    for (const scan of session.scans) {
      if (images.current.has(scan.file)) continue;
      const img = new Image();
      img.src = `/ctg/${session.id}/scans/${scan.file}`;
      img.onload = () => {
        if (!alive) return;
        images.current.set(scan.file, img);
        drawRef.current();
      };
    }
    return () => {
      alive = false;
    };
  }, [session]);

  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, pxPerCm, showPhoto, photoOpacity, shiftSec, showFhr, showMhr, showToco, activeScan, width, height]);

  useEffect(() => {
    if (scrollToSec == null || !scrollRef.current) return;
    const x = 12 + (scrollToSec / 60) * pxPerCm;
    scrollRef.current.scrollTo({ left: Math.max(0, x - 40), behavior: "smooth" });
  }, [scrollToSec, pxPerCm]);

  function xOf(t: number) {
    return 12 + (t / 60) * pxPerCm;
  }
  function yFhr(bpm: number) {
    return fhrY + ((FHR_TOP - bpm) / 20) * pxPerCm;
  }
  function yToco(kpa: number) {
    return tocoY + ((TOCO_TOP - kpa) / 2) * pxPerCm;
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#f3f1e7";
    ctx.fillRect(0, 0, width, height);

    const grid = "rgba(109, 138, 98, 0.45)";
    const gridStrong = "rgba(109, 138, 98, 0.9)";

    // FHR grid. 1 mm = 0.1 cm = 2 bpm = 6 s
    for (let bpm = FHR_BOT; bpm <= FHR_TOP; bpm += 2) {
      const y = yFhr(bpm);
      ctx.beginPath();
      ctx.strokeStyle = bpm % 20 === 0 ? gridStrong : grid;
      ctx.lineWidth = bpm % 20 === 0 ? 1 : 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    for (let kpa = 0; kpa <= 12; kpa += 0.2) {
      const y = yToco(round1(kpa));
      const major = Math.abs(kpa % 2) < 0.01;
      ctx.beginPath();
      ctx.strokeStyle = major ? gridStrong : grid;
      ctx.lineWidth = major ? 1 : 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    const minutes = session.durationSec / 60;
    for (let m = 0; m <= minutes; m += 0.1) {
      const x = xOf(m * 60);
      const major = Math.abs(m - Math.round(m)) < 0.001;
      ctx.beginPath();
      ctx.strokeStyle = major ? gridStrong : grid;
      ctx.lineWidth = major ? 1 : 0.5;
      ctx.moveTo(x, fhrY);
      ctx.lineTo(x, fhrY + fhrH);
      ctx.moveTo(x, tocoY);
      ctx.lineTo(x, tocoY + tocoH);
      ctx.stroke();
    }

    ctx.fillStyle = "#f3f1e7";
    ctx.fillRect(0, fhrY + fhrH, width, gapCm * pxPerCm);

    if (showPhoto && activeScan) {
      const scan = session.scans.find((s) => s.file === activeScan);
      const img = scan ? images.current.get(scan.file) : undefined;
      if (scan && img) {
        ctx.save();
        ctx.globalAlpha = photoOpacity;
        const x0 = xOf(scan.t0 + shiftSec);
        const x1 = xOf(scan.t1 + shiftSec);
        const dw = x1 - x0;
        // Frontal shots: time left to right. 200–60 is the labeled FHR grid,
        // 12–0 the toco grid. Fractions measured on the upright 12-22 set.
        if (dw > 4) {
          ctx.drawImage(img, 0, img.height * 0.105, img.width, img.height * 0.4, x0, yFhr(200), dw, yFhr(60) - yFhr(200));
          ctx.drawImage(img, 0, img.height * 0.56, img.width, img.height * 0.3, x0, yToco(12), dw, yToco(0) - yToco(12));
        }
        ctx.restore();
      }
    }

    if (showFhr) stroke(ctx, session.fhr, session.dt, yFhr, "#1c1914", 1.25);
    if (showMhr) stroke(ctx, session.mhr, session.dt, yFhr, "#c4622d", 1.05);
    if (showToco) stroke(ctx, session.toco, session.dt, yToco, "#1c1914", 1.15);

    ctx.font = "11px IBM Plex Sans, sans-serif";
    ctx.fillStyle = "#1c1914";
    let lastLabel = -999;
    for (const ev of session.events) {
      const x = xOf(ev.t);
      if (x - lastLabel < 108) continue;
      lastLabel = x;
      const short =
        ev.kind === "puls"
          ? `Puls ${ev.value ?? ""}`
          : ev.kind === "kineto"
            ? `Kineto ${ev.value ?? ""}%`
            : ev.text;
      ctx.fillText(short, x + 2, tocoY - 5);
    }

    ctx.font = "11px IBM Plex Mono, monospace";
    ctx.fillStyle = "#1c1914";
    for (let m = 0; m <= minutes; m += 5) {
      const label = clockOf(session.startedAt, m * 60).slice(0, 5);
      ctx.fillText(label, xOf(m * 60) + 3, height - 6);
    }
  }

  function stroke(
    ctx: CanvasRenderingContext2D,
    series: number[],
    dt: number,
    yMap: (v: number) => number,
    color: string,
    widthPx: number,
  ) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = widthPx;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    let started = false;
    for (let i = 0; i < series.length; i++) {
      const v = series[i];
      const x = xOf(i * dt);
      const y = yMap(v);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function onMove(clientX: number) {
    const canvas = canvasRef.current;
    const sc = scrollRef.current;
    if (!canvas || !sc) return;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const t = ((x - 12) / pxPerCm) * 60;
    if (t < 0 || t > session.durationSec) onCursor(null);
    else onCursor(t);
  }

  const scale = (
    <div className="relative w-11 shrink-0" style={{ height }}>
      {scaleTicks(FHR_TOP, FHR_BOT, 20, fhrY, fhrH, (v) => String(v))}
      {scaleTicks(TOCO_TOP, TOCO_BOT, 2, tocoY, tocoH, (v) => String(v))}
    </div>
  );

  return (
    <div className="flex min-h-0 flex-1">
      {scale}
      <div
        ref={scrollRef}
        className="min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
        onScroll={() => {
          /* keeps the native scrollbar; minute rail reads scroll via parent if needed */
        }}
      >
        <canvas
          ref={canvasRef}
          onPointerMove={(e) => onMove(e.clientX)}
          onPointerLeave={() => onCursor(null)}
          className="block touch-pan-x"
        />
      </div>
    </div>
  );
}

function scaleTicks(
  top: number,
  bot: number,
  step: number,
  y0: number,
  h: number,
  fmt: (v: number) => string,
) {
  const ticks = [];
  for (let v = bot; v <= top; v += step) {
    const y = y0 + ((top - v) / (top - bot)) * h;
    ticks.push(
      <span
        key={v}
        className="absolute right-1 -translate-y-1/2 font-mono text-xs text-paper"
        style={{ top: y, fontSize: 10 }}
      >
        {fmt(v)}
      </span>,
    );
  }
  return ticks;
}

function round1(n: number) {
  return Math.round(n * 10) / 10;
}
