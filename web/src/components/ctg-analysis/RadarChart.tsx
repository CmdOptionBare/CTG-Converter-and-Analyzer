import { useState } from "react";
import type { RadarScores } from "@/lib/ctg-analysis/metrics";
import { formatDe } from "@/lib/ctg-analysis/format";

const AXES: { key: keyof RadarScores; label: string; full: string }[] = [
  { key: "baselineStabilitaet", label: "Basis", full: "Baseline-Stabilität" },
  { key: "variabilitaet", label: "Variabilität", full: "Variabilität aus dem IQR" },
  { key: "komplexitaet", label: "Komplexität", full: "Komplexität, hohes Gzip-Verhältnis" },
  { key: "dezelerationsLast", label: "Last", full: "Dezelerations-Last" },
  { key: "erholungsrate", label: "Erholung", full: "Erholungsrate" },
  { key: "kopplung", label: "Kopplung", full: "FHR–Toco-Kopplung" },
];

const CX = 210;
const CY = 168;
const RADIUS = 108;

export function RadarChart({ scores }: { scores: RadarScores }) {
  const [active, setActive] = useState<number | null>(null);
  const polygon = AXES.map((axis, index) => {
    const value = scores[axis.key];
    return point(index, RADIUS * ((value ?? 0) / 100));
  }).join(" ");
  const shown = active == null ? null : AXES[active];
  const shownScore = shown ? scores[shown.key] : null;

  return (
    <svg viewBox="0 0 420 340" className="mx-auto block h-auto w-full max-w-md" role="img" aria-label="Radar der sechs Kennzahlen">
      {[0.25, 0.5, 0.75, 1].map((step) => (
        <polygon
          key={step}
          points={AXES.map((_, index) => point(index, RADIUS * step)).join(" ")}
          fill="none"
          stroke="#2a332d"
        />
      ))}
      {AXES.map((axis, index) => {
        const [x, y] = pair(index, RADIUS);
        const [lx, ly] = pair(index, RADIUS + 28);
        return (
          <g key={axis.key} onPointerEnter={() => setActive(index)} onPointerLeave={() => setActive(null)}>
            <line x1={CX} y1={CY} x2={x} y2={y} stroke="#2a332d" />
            <text
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="middle"
              fill={active === index ? "#c4622d" : "rgba(243,241,231,0.72)"}
              fontSize="12"
              fontFamily="IBM Plex Sans, sans-serif"
            >
              {axis.label}
            </text>
          </g>
        );
      })}
      <polygon points={polygon} fill="rgba(196,98,45,0.28)" stroke="#c4622d" strokeWidth="1.5" />
      {AXES.map((axis, index) => {
        const value = scores[axis.key];
        const [x, y] = pair(index, RADIUS * ((value ?? 0) / 100));
        return <circle key={axis.key} cx={x} cy={y} r={active === index ? 4 : 2.5} fill="#f3f1e7" />;
      })}
      {shown ? (
        <>
          <text x={CX} y={CY - 6} textAnchor="middle" fill="#f3f1e7" fontSize="12" fontFamily="IBM Plex Sans, sans-serif">
            {shown.full}
          </text>
          <text x={CX} y={CY + 16} textAnchor="middle" fill="#c4622d" fontSize="16" fontFamily="IBM Plex Mono, monospace">
            {formatDe(shownScore, 0)}
          </text>
        </>
      ) : null}
    </svg>
  );
}

function pair(index: number, radius: number): [number, number] {
  const angle = -Math.PI / 2 + (index * 2 * Math.PI) / AXES.length;
  return [CX + Math.cos(angle) * radius, CY + Math.sin(angle) * radius];
}

function point(index: number, radius: number): string {
  const [x, y] = pair(index, radius);
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}
