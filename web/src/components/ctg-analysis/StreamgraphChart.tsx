import { useMemo, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { CtgSample } from "@/lib/ctg/types";
import { isSample } from "@/lib/ctg-analysis/metrics";
import { clockOf, formatDe } from "@/lib/ctg-analysis/format";
import { HoverTip } from "./ChartFrame";

const ZONES = [
  { label: "< 110", color: "#c4622d" },
  { label: "110–160", color: "#d9d3c3" },
  { label: "> 160", color: "#6d8a62" },
  { label: "Lücke", color: "#3c463f" },
];

const BLOCK = 300;

type Block = { t0: number; t1: number; counts: number[]; shares: number[] };

export function StreamgraphChart({
  fhr,
  dt,
  durationSec,
  startedAt,
}: {
  fhr: CtgSample[];
  dt: number;
  durationSec: number;
  startedAt: string;
}) {
  const blocks = useMemo(() => buildBlocks(fhr, dt, durationSec), [fhr, dt, durationSec]);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const width = 1000;
  const height = 200;
  const pad = { l: 8, r: 8, t: 12, b: 28 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;

  function xOf(t: number): number {
    if (!(durationSec > 0)) return pad.l;
    return pad.l + (t / durationSec) * plotW;
  }
  function yOf(shareFromCenter: number): number {
    return pad.t + (0.5 - shareFromCenter) * plotH;
  }

  const edges = blocks.map((block) => {
    let acc = -0.5;
    const ys = [acc];
    for (const share of block.shares) {
      acc += share;
      ys.push(acc);
    }
    return ys;
  });

  function move(event: ReactPointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * width;
    const t = durationSec > 0 ? ((x - pad.l) / plotW) * durationSec : 0;
    const block = blocks.find((item) => t >= item.t0 && t < item.t1) ?? blocks[blocks.length - 1];
    if (!block) {
      setTip(null);
      return;
    }
    const lines = ZONES.map((zone, index) => `${zone.label}  ${formatDe(block.counts[index], 0)} s`).join("\n");
    setTip({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      text: `${clockOf(startedAt, block.t0).slice(0, 5)}–${clockOf(startedAt, block.t1).slice(0, 5)}\n${lines}`,
    });
  }

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block h-auto w-full"
        role="img"
        aria-label="Zeitanteile in Herzfrequenz-Zonen je fünf Minuten"
        onPointerMove={move}
        onPointerLeave={() => setTip(null)}
      >
        {ZONES.map((zone, zoneIndex) => (
          <path key={zone.label} d={areaPath(blocks, edges, zoneIndex, xOf, yOf)} fill={zone.color} opacity={zoneIndex === 3 ? 0.9 : 0.92} />
        ))}
        {blocks.map((block) => (
          <text
            key={block.t0}
            x={xOf((block.t0 + block.t1) / 2)}
            y={height - 8}
            textAnchor="middle"
            fill="rgba(243,241,231,0.4)"
            fontSize="11"
            fontFamily="IBM Plex Mono, monospace"
          >
            {block.t0 % 600 === 0 ? clockOf(startedAt, block.t0).slice(0, 5) : ""}
          </text>
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-paper/70">
        {ZONES.map((zone) => (
          <span key={zone.label} className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2 w-4" style={{ background: zone.color }} />
            {zone.label}
          </span>
        ))}
      </div>
      {tip ? <HoverTip x={tip.x} y={tip.y} text={tip.text} /> : null}
    </div>
  );
}

function buildBlocks(fhr: CtgSample[], dt: number, durationSec: number): Block[] {
  const out: Block[] = [];
  if (!(durationSec > 0) || !(dt > 0)) return out;
  for (let t0 = 0; t0 < durationSec; t0 += BLOCK) {
    const t1 = Math.min(durationSec, t0 + BLOCK);
    const i0 = Math.max(0, Math.floor(t0 / dt));
    const i1 = Math.min(fhr.length, Math.ceil(t1 / dt));
    const counts = [0, 0, 0, 0];
    for (let i = i0; i < i1; i++) {
      const value = fhr[i];
      if (!isSample(value)) counts[3] += 1;
      else if (value < 110) counts[0] += 1;
      else if (value <= 160) counts[1] += 1;
      else counts[2] += 1;
    }
    const total = counts.reduce((sum, value) => sum + value, 0);
    out.push({
      t0,
      t1,
      counts,
      shares: total ? counts.map((value) => value / total) : [0, 0, 0, 0],
    });
  }
  return out;
}

function areaPath(
  blocks: Block[],
  edges: number[][],
  zone: number,
  xOf: (t: number) => number,
  yOf: (share: number) => number,
): string {
  if (!blocks.length) return "";
  let d = "";
  blocks.forEach((block, index) => {
    const x = xOf((block.t0 + block.t1) / 2);
    const y = yOf(edges[index][zone + 1]);
    d += index === 0 ? `M ${x.toFixed(2)} ${y.toFixed(2)}` : ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  for (let index = blocks.length - 1; index >= 0; index--) {
    const block = blocks[index];
    const x = xOf((block.t0 + block.t1) / 2);
    const y = yOf(edges[index][zone]);
    d += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  }
  return `${d} Z`;
}
