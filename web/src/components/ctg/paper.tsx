import { useEffect, useMemo, useRef } from "react";
import type { CtgSample, CtgSession } from "@/lib/ctg/types";
import { clockOf, isSample } from "@/lib/ctg/analyze";

type Props = {
  session: CtgSession;
  pxPerCm: number;
  showPhoto: boolean;
  photoOpacity: number;
  showFhr: boolean;
  showMhr: boolean;
  showToco: boolean;
  onCursor: (t: number | null) => void;
  scrollToSec?: number;
  scrollToken?: number;
};

type Layout = {
  left: number;
  top: number;
  scale: number;
  imgW: number;
  imgH: number;
  width: number;
  height: number;
  fhrTopY: number;
  fhrBotY: number;
  tocoTopY: number;
  tocoBotY: number;
  nFhr: number;
  nToco: number;
  fhrLabelStep: number;
  tocoLabelStep: number;
  verticalMinorSec: number;
  verticalMajorEvery: number;
};

function layoutOf(session: CtgSession, pxPerCm: number): Layout {
  const pano = session.panorama;
  const left = 1;
  const top = 0;
  const padRight = 12;
  const padBot = 18;
  const scale = pxPerCm / pano.pxPerCm;
  const imgW = pano.widthPx * scale;
  const imgH = pano.heightPx * scale;
  const yAt = (yPx: number) => top + yPx * scale;
  const nFhr = Math.max(1, Math.round(pano.fhrHeightCm / pano.fhrMinorCm));
  const nToco = Math.max(1, Math.round(pano.tocoHeightCm / pano.tocoMinorCm));
  const fhrLabelStep = ((pano.fhrTopBpm - pano.fhrBotBpm) / nFhr) * Math.round(1 / pano.fhrMinorCm);
  const tocoMinor = (pano.tocoTop - pano.tocoBot) / nToco;
  const tocoLabelStep = tocoMinor * 2;
  return {
    left,
    top,
    scale,
    imgW,
    imgH,
    width: Math.ceil(left + imgW + padRight),
    height: Math.ceil(top + imgH + padBot),
    fhrTopY: yAt(pano.fhr210YPx),
    fhrBotY: yAt(pano.fhr50YPx),
    tocoTopY: yAt(pano.toco100YPx),
    tocoBotY: yAt(pano.toco0YPx),
    nFhr,
    nToco,
    fhrLabelStep,
    tocoLabelStep,
    verticalMinorSec: (pano.verticalMinorCm / session.speedCmMin) * 60,
    verticalMajorEvery: Math.max(1, Math.round(1 / pano.verticalMinorCm)),
  };
}

export function CtgPaper({
  session,
  pxPerCm,
  showPhoto,
  photoOpacity,
  showFhr,
  showMhr,
  showToco,
  onCursor,
  scrollToSec,
  scrollToken,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const layout = useMemo(() => layoutOf(session, pxPerCm), [session, pxPerCm]);
  const pano = session.panorama;

  const drawRef = useRef<() => void>(() => {});

  function xOf(t: number) {
    return layout.left + (t / 60) * session.speedCmMin * pxPerCm;
  }
  function yFhr(bpm: number) {
    const t = (pano.fhrTopBpm - bpm) / (pano.fhrTopBpm - pano.fhrBotBpm);
    return layout.fhrTopY + t * (layout.fhrBotY - layout.fhrTopY);
  }
  function yToco(value: number) {
    const t = (pano.tocoTop - value) / (pano.tocoTop - pano.tocoBot);
    return layout.tocoTopY + t * (layout.tocoBotY - layout.tocoTopY);
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.floor(layout.width * dpr);
    canvas.height = Math.floor(layout.height * dpr);
    canvas.style.width = `${layout.width}px`;
    canvas.style.height = `${layout.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#f3f1e7";
    ctx.fillRect(0, 0, layout.width, layout.height);

    const photo = imageRef.current;
    if (showPhoto && photo && photo.complete && photo.naturalWidth > 0) {
      ctx.save();
      ctx.globalAlpha = photoOpacity;
      ctx.drawImage(photo, layout.left, layout.top, layout.imgW, layout.imgH);
      ctx.restore();
    }

    const grid = "rgba(109, 138, 98, 0.45)";
    const gridStrong = "rgba(109, 138, 98, 0.9)";
    const xRight = layout.left + layout.imgW;

    for (let i = 0; i <= layout.nFhr; i++) {
      const bpm = pano.fhrTopBpm + (i / layout.nFhr) * (pano.fhrBotBpm - pano.fhrTopBpm);
      const major = i % layout.verticalMajorEvery === 0;
      hLine(ctx, yFhr(bpm), layout.left, xRight, major ? gridStrong : grid, major ? 1 : 0.5);
    }
    for (let i = 0; i <= layout.nToco; i++) {
      const value = pano.tocoTop + (i / layout.nToco) * (pano.tocoBot - pano.tocoTop);
      const major = i === 0 || i === layout.nToco;
      hLine(ctx, yToco(value), layout.left, xRight, major ? gridStrong : grid, major ? 1 : 0.5);
    }

    const imageDurationSec = (pano.widthPx / pano.pxPerCm / session.speedCmMin) * 60;
    const nVert = Math.floor(imageDurationSec / layout.verticalMinorSec + 1e-6);
    for (let i = 0; i <= nVert; i++) {
      const x = xOf(i * layout.verticalMinorSec);
      if (x < layout.left - 0.5 || x > xRight + 0.5) continue;
      const major = i % layout.verticalMajorEvery === 0;
      ctx.beginPath();
      ctx.strokeStyle = major ? gridStrong : grid;
      ctx.lineWidth = major ? 1 : 0.5;
      ctx.moveTo(x, layout.fhrTopY);
      ctx.lineTo(x, layout.fhrBotY);
      ctx.moveTo(x, layout.tocoTopY);
      ctx.lineTo(x, layout.tocoBotY);
      ctx.stroke();
    }

    const halo = !showPhoto;
    if (showFhr) strokeSeries(ctx, session.fhr, session.dt, yFhr, "#1c1914", 1.25, halo);
    if (showMhr) strokeSeries(ctx, session.mhr, session.dt, yFhr, "#c4622d", 1.05, halo);
    if (showToco) strokeSeries(ctx, session.tocoPct, session.dt, yToco, "#1c1914", 1.15, halo);

    ctx.font = "11px IBM Plex Sans, sans-serif";
    ctx.fillStyle = "#1c1914";
    let lastLabel = -999;
    for (const ev of session.events) {
      const x = xOf(ev.t);
      if (x - lastLabel < 108) continue;
      lastLabel = x;
      const short =
        ev.kind === "puls" && ev.value != null
          ? `Puls ${ev.value}`
          : ev.kind === "kineto" && ev.value != null
            ? `Kineto ${ev.value}%`
            : ev.text;
      ctx.fillText(short, x + 2, layout.tocoTopY - 5);
    }

    ctx.font = "11px IBM Plex Mono, monospace";
    ctx.fillStyle = "#1c1914";
    const minutes = session.durationSec / 60;
    for (let m = 0; m <= minutes; m += 5) {
      const label = clockOf(session.startedAt, m * 60).slice(0, 5);
      ctx.fillText(label, xOf(m * 60) + 3, layout.height - 6);
    }
  }

  function strokeSeries(
    ctx: CanvasRenderingContext2D,
    series: CtgSample[],
    dt: number,
    yMap: (v: number) => number,
    color: string,
    widthPx: number,
    halo: boolean,
  ) {
    if (halo) strokePath(ctx, series, dt, yMap, "rgba(243, 241, 231, 0.92)", widthPx + 1.15);
    strokePath(ctx, series, dt, yMap, color, widthPx);
  }

  function strokePath(
    ctx: CanvasRenderingContext2D,
    series: CtgSample[],
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
    let penDown = false;
    for (let i = 0; i < series.length; i++) {
      const v = series[i];
      if (!isSample(v)) {
        penDown = false;
        continue;
      }
      const x = xOf(i * dt);
      const y = yMap(v);
      if (!penDown) {
        ctx.moveTo(x, y);
        penDown = true;
      } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  drawRef.current = draw;

  useEffect(() => {
    let alive = true;
    const img = new Image();
    img.src = `/ctg/${session.id}/${pano.file}`;
    img.onload = () => {
      if (!alive) return;
      imageRef.current = img;
      drawRef.current();
    };
    return () => {
      alive = false;
    };
  }, [session.id, pano.file]);

  useEffect(() => {
    draw();
    // Drawing reads the latest image via a ref; the effect reruns when the view changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, pxPerCm, showPhoto, photoOpacity, showFhr, showMhr, showToco, layout]);

  useEffect(() => {
    if (scrollToSec == null || !scrollRef.current) return;
    const x = xOf(scrollToSec);
    scrollRef.current.scrollTo({ left: Math.max(0, x - 40), behavior: "smooth" });
    // xOf is rebuilt with pxPerCm and the session speed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scrollToSec, scrollToken, pxPerCm, session.speedCmMin]);

  function onMove(clientX: number) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const t = ((x - layout.left) / pxPerCm / session.speedCmMin) * 60;
    const end = (pano.widthPx / pano.pxPerCm / session.speedCmMin) * 60;
    if (t < 0 || t > end) onCursor(null);
    else onCursor(t);
  }

  const scale = (
    <div className="relative w-11 shrink-0" style={{ height: layout.height }}>
      {scaleTicks(pano.fhrBotBpm, pano.fhrTopBpm, layout.fhrLabelStep, yFhr, (v) => String(Math.round(v)))}
      {scaleTicks(pano.tocoBot, pano.tocoTop, layout.tocoLabelStep, yToco, (v) => String(Math.round(v)))}
    </div>
  );

  return (
    <div className="flex min-h-0 flex-1">
      {scale}
      <div ref={scrollRef} className="min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
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

function hLine(
  ctx: CanvasRenderingContext2D,
  y: number,
  x0: number,
  x1: number,
  color: string,
  widthPx: number,
) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = widthPx;
  ctx.moveTo(x0, y);
  ctx.lineTo(x1, y);
  ctx.stroke();
}

function scaleTicks(
  bot: number,
  top: number,
  step: number,
  yOf: (v: number) => number,
  fmt: (v: number) => string,
) {
  if (!(step > 0)) return null;
  const ticks = [];
  const n = Math.round((top - bot) / step);
  for (let i = 0; i <= n; i++) {
    const v = bot + i * step;
    ticks.push(
      <span
        key={i}
        className="absolute right-1 -translate-y-1/2 font-mono text-paper"
        style={{ top: yOf(v), fontSize: 10 }}
      >
        {fmt(v)}
      </span>,
    );
  }
  return ticks;
}
