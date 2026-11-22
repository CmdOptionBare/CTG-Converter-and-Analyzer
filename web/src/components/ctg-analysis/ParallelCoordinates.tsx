import { useState } from "react";
import type { Deceleration } from "@/lib/ctg-analysis/detectors";
import { formatDe, formatSigned } from "@/lib/ctg-analysis/format";

const AXES: { key: keyof Deceleration; label: string; min: number; max: number; digits: number; signed?: boolean }[] = [
  { key: "durationSec", label: "Dauer", min: 0, max: 150, digits: 0 },
  { key: "depth", label: "Tiefe", min: 0, max: 60, digits: 0 },
  { key: "auc", label: "AUC", min: 0, max: 3600, digits: 0 },
  { key: "lagSec", label: "Lag", min: -90, max: 60, digits: 0, signed: true },
  { key: "recoverySlope", label: "Erholung", min: -0.5, max: 1.6, digits: 2, signed: true },
  { key: "gzipRatio", label: "Gzip", min: 0.4, max: 1.2, digits: 2 },
];

const PAD_X = 54;
const HEIGHT = 280;

export function ParallelCoordinates({
  events,
  selected,
  onSelect,
}: {
  events: Deceleration[];
  selected: number;
  onSelect?: (index: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  if (!events.length) {
    return <p className="text-sm text-paper/60">Keine Dezeleration von mindestens 15 Sekunden.</p>;
  }
  const width = 720;
  const plotW = width - PAD_X * 2;
  const top = 28;
  const bottom = HEIGHT - 36;

  function xOf(index: number): number {
    return PAD_X + (index / (AXES.length - 1)) * plotW;
  }
  function yOf(axis: (typeof AXES)[number], value: number): number {
    const t = (Math.min(axis.max, Math.max(axis.min, value)) - axis.min) / (axis.max - axis.min);
    return bottom - t * (bottom - top);
  }

  return (
    <svg viewBox={`0 0 ${width} ${HEIGHT}`} className="block h-auto w-full" role="img" aria-label="Parallelkoordinaten der Dezelerationen">
      {AXES.map((axis, index) => (
        <g key={axis.key}>
          <line x1={xOf(index)} y1={top} x2={xOf(index)} y2={bottom} stroke="#2a332d" />
          <text x={xOf(index)} y={16} textAnchor="middle" fill="rgba(243,241,231,0.7)" fontSize="12" fontFamily="IBM Plex Sans, sans-serif">
            {axis.label}
          </text>
          <text x={xOf(index)} y={HEIGHT - 12} textAnchor="middle" fill="rgba(243,241,231,0.35)" fontSize="10" fontFamily="IBM Plex Mono, monospace">
            {axis.signed ? formatSigned(axis.min, axis.digits) : formatDe(axis.min, axis.digits)}
            {" · "}
            {axis.signed ? formatSigned(axis.max, axis.digits) : formatDe(axis.max, axis.digits)}
          </text>
        </g>
      ))}
      {events.map((event, index) => {
        const active = index === selected || index === hover;
        const stroke = index === selected ? "#c4622d" : "rgba(243,241,231,0.35)";
        return (
          <g
            key={`${event.t0}-${index}`}
            onPointerEnter={() => setHover(index)}
            onPointerLeave={() => setHover(null)}
            onClick={() => onSelect?.(index)}
            className="cursor-pointer"
          >
            {segments(event, xOf, yOf).map((points, part) => (
              <g key={part}>
                <polyline points={points} fill="none" stroke="transparent" strokeWidth="10" />
                <polyline points={points} fill="none" stroke={stroke} strokeWidth={active ? 2.2 : 1} />
              </g>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

function segments(
  event: Deceleration,
  xOf: (index: number) => number,
  yOf: (axis: (typeof AXES)[number], value: number) => number,
): string[] {
  const out: string[] = [];
  let current: string[] = [];
  AXES.forEach((axis, index) => {
    const value = event[axis.key];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      if (current.length > 1) out.push(current.join(" "));
      current = [];
      return;
    }
    current.push(`${xOf(index).toFixed(1)},${yOf(axis, value).toFixed(1)}`);
  });
  if (current.length > 1) out.push(current.join(" "));
  return out;
}
