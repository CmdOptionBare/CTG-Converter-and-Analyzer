import type { CtgSession } from "@/lib/ctg/types";

export type WindowStats = {
  t0: number;
  t1: number;
  baseline: number;
  low: number;
  high: number;
  band: number;
  rises: number;
  falls: number;
  mhr: number;
  toco: number;
};

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function percentile(xs: number[], p: number): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[i];
}

function episodes(values: number[], dt: number, pred: (v: number) => boolean, minSec: number): number {
  let run = 0;
  let n = 0;
  for (const v of values) {
    if (pred(v)) run += dt;
    else {
      if (run >= minSec) n += 1;
      run = 0;
    }
  }
  if (run >= minSec) n += 1;
  return n;
}

export function windowStats(session: CtgSession, winSec = 600): WindowStats[] {
  const out: WindowStats[] = [];
  const { dt, fhr, mhr, toco, durationSec } = session;
  for (let t0 = 0; t0 < durationSec; t0 += winSec) {
    const t1 = Math.min(durationSec, t0 + winSec);
    const i0 = Math.floor(t0 / dt);
    const i1 = Math.min(fhr.length, Math.ceil(t1 / dt));
    const f = fhr.slice(i0, i1);
    const m = mhr.slice(i0, i1);
    const c = toco.slice(i0, i1);
    const baseline = median(f);
    const low = percentile(f, 10);
    const high = percentile(f, 90);
    out.push({
      t0,
      t1,
      baseline: Math.round(baseline),
      low: Math.round(low),
      high: Math.round(high),
      band: Math.round(high - low),
      rises: episodes(f, dt, (v) => v >= baseline + 15, 15),
      falls: episodes(f, dt, (v) => v <= baseline - 15, 15),
      mhr: Math.round(median(m)),
      toco: Math.round(median(c) * 10) / 10,
    });
  }
  return out;
}

export function clockOf(startedAt: string, tSec: number): string {
  const [h, m, s] = startedAt.slice(11, 19).split(":").map(Number);
  const total = h * 3600 + m * 60 + s + Math.max(0, Math.round(tSec));
  const hh = Math.floor(total / 3600) % 24;
  const mm = Math.floor(total / 60) % 60;
  const ss = total % 60;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(hh)}:${p(mm)}:${p(ss)}`;
}

export function toCsv(session: CtgSession): string {
  const lines = ["t_sec,clock,fhr_bpm,mhr_bpm,toco_kpa"];
  for (let i = 0; i < session.fhr.length; i++) {
    const t = i * session.dt;
    lines.push(
      [t, clockOf(session.startedAt, t), session.fhr[i], session.mhr[i], session.toco[i]].join(","),
    );
  }
  return lines.join("\n");
}
