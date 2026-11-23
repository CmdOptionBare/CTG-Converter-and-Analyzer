import { useMemo, useState } from "react";
import type { CtgEvent } from "@/lib/ctg/types";
import type { Deceleration } from "@/lib/ctg-analysis/detectors";
import type { TimeRange } from "@/lib/ctg-analysis/windows";
import { clockOf, formatDe } from "@/lib/ctg-analysis/format";
import { HoverTip } from "./ChartFrame";

type Mark = { t: number; t0: number; t1: number; lane: number; text: string; color: string };

type Props = {
  durationSec: number;
  startedAt: string;
  decels: Deceleration[];
  gaps: TimeRange[];
  events: CtgEvent[];
  cursorSec?: number | null;
  activeT?: number | null;
};

const LANES = ["Dezeleration", "Lücke", "Kineto"];

export function BarcodePlot({ durationSec, startedAt, decels, gaps, events, cursorSec, activeT }: Props) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const marks = useMemo(() => buildMarks(decels, gaps, events, startedAt), [decels, gaps, events, startedAt]);

  function xOf(t: number): string {
    if (!(durationSec > 0)) return "0%";
    return `${(Math.min(1, Math.max(0, t / durationSec)) * 100).toFixed(4)}%`;
  }

  return (
    <div className="relative">
      <div className="space-y-1">
        {LANES.map((label, lane) => (
          <div key={label} className="grid grid-cols-[7.5rem_1fr] items-center gap-2">
            <div className="text-xs text-paper/50">{label}</div>
            <div
              className="relative h-8 rounded-card bg-raised"
              onPointerMove={(event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const t = durationSec > 0 ? (x / rect.width) * durationSec : 0;
                const hit = nearest(marks.filter((mark) => mark.lane === lane), t, durationSec, rect.width);
                setTip(
                  hit
                    ? { x: event.clientX - (event.currentTarget.parentElement?.getBoundingClientRect().left ?? 0), y: 8, text: hit.text }
                    : null,
                );
              }}
              onPointerLeave={() => setTip(null)}
            >
              {marks
                .filter((mark) => mark.lane === lane)
                .map((mark, index) => (
                  <span key={`${mark.t}-${index}`}>
                    <span
                      className="absolute top-3 h-2 opacity-40"
                      style={{
                        left: xOf(mark.t0),
                        width: `max(1px, calc(${xOf(mark.t1)} - ${xOf(mark.t0)}))`,
                        background: mark.color,
                      }}
                    />
                    <span
                      className="absolute top-1 h-6 w-px"
                      style={{ left: xOf(mark.t), background: mark.color }}
                    />
                  </span>
                ))}
              {cursorSec != null ? (
                <span className="absolute top-0 h-full w-px bg-paper/70" style={{ left: xOf(cursorSec) }} />
              ) : null}
              {activeT != null ? (
                <span className="absolute top-0 h-full w-0.5 bg-copper" style={{ left: xOf(activeT) }} />
              ) : null}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-[7.5rem_1fr] text-[11px] text-paper/40">
        <span />
        <div className="flex justify-between font-mono">
          <span>{clockOf(startedAt, 0).slice(0, 5)}</span>
          <span>{clockOf(startedAt, durationSec / 2).slice(0, 5)}</span>
          <span>{clockOf(startedAt, durationSec).slice(0, 5)}</span>
        </div>
      </div>
      {tip ? <HoverTip x={tip.x} y={tip.y} text={tip.text} /> : null}
    </div>
  );
}

function buildMarks(decels: Deceleration[], gaps: TimeRange[], events: CtgEvent[], startedAt: string): Mark[] {
  const marks: Mark[] = [];
  for (const event of decels) {
    marks.push({
      t: event.troughT,
      t0: event.t0,
      t1: event.t1,
      lane: 0,
      color: "#c4622d",
      text: `${clockOf(startedAt, event.troughT)}\nTiefe ${formatDe(event.depth, 0)} bpm\n${formatDe(event.durationSec, 0)} s`,
    });
  }
  for (const gap of gaps) {
    marks.push({
      t: (gap.t0 + gap.t1) / 2,
      t0: gap.t0,
      t1: gap.t1,
      lane: 1,
      color: "#8d887c",
      text: `Lücke ${formatDe(gap.t1 - gap.t0, 0)} s`,
    });
  }
  for (const event of events) {
    if (event.kind !== "kineto") continue;
    marks.push({
      t: event.t,
      t0: event.t,
      t1: event.t + 1,
      lane: 2,
      color: "#6d8a62",
      text: event.text || "Kineto",
    });
  }
  return marks;
}

function nearest(marks: Mark[], t: number, duration: number, width: number): Mark | null {
  if (!marks.length || !(width > 0)) return null;
  const tol = (8 / width) * duration;
  let best: Mark | null = null;
  let bestDist = Infinity;
  for (const mark of marks) {
    const dist = Math.abs(mark.t - t);
    if (dist < bestDist) {
      bestDist = dist;
      best = mark;
    }
  }
  return best && bestDist <= Math.max(tol, 4) ? best : null;
}
