import { useEffect, useState, type RefObject } from "react";

export const palette = {
  bg: "#141816",
  raised: "#1c221e",
  paper: "#f3f1e7",
  copper: "#c4622d",
  grid: "#6d8a62",
  line: "#2a332d",
  ink: "#1c1914",
};

export function useElementWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const apply = () => setWidth(el.clientWidth);
    apply();
    const observer = new ResizeObserver(apply);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

export function setupCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D | null {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function mix(hexA: string, hexB: string, t: number): string {
  const a = hexToRgb(hexA);
  const b = hexToRgb(hexB);
  const u = Math.min(1, Math.max(0, t));
  const c = a.map((channel, i) => Math.round(channel + (b[i] - channel) * u));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
