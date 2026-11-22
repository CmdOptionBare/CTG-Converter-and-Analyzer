import type { CtgSample, CtgSession } from "@/lib/ctg/types";

export type WindowStats = {
  t0: number;
  t1: number;
  baseline: number | null;
  low: number | null;
  high: number | null;
  band: number | null;
  rises: number;
  falls: number;
  mhr: number | null;
  toco: number | null;
};

export function isSample(v: CtgSample | undefined): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function formatSample(v: CtgSample | undefined, digits = 0): string {
  if (!isSample(v)) return "–";
  if (digits <= 0) return String(Math.round(v));
  return v.toFixed(digits);
}

/** kPa derived from the 0–100 scale. Null when the factor or the sample is missing. */
export function tocoKpa(pct: CtgSample | undefined, factor: number | undefined): number | null {
  if (!isSample(pct) || factor == null || !Number.isFinite(factor)) return null;
  return Math.round(pct * factor * 10) / 10;
}

function finite(xs: CtgSample[]): number[] {
  const out: number[] = [];
  for (const v of xs) {
    if (isSample(v)) out.push(v);
  }
  return out;
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function percentile(xs: number[], p: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[i];
}

function roundOrNull(v: number | null, digits = 0): number | null {
  if (v == null || !Number.isFinite(v)) return null;
  const p = 10 ** digits;
  return Math.round(v * p) / p;
}

function episodes(values: CtgSample[], dt: number, pred: (v: number) => boolean, minSec: number): number {
  let run = 0;
  let n = 0;
  for (const v of values) {
    if (isSample(v) && pred(v)) run += dt;
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
  const { dt, fhr, mhr, tocoPct, durationSec } = session;
  for (let t0 = 0; t0 < durationSec; t0 += winSec) {
    const t1 = Math.min(durationSec, t0 + winSec);
    const i0 = Math.floor(t0 / dt);
    const i1 = Math.min(fhr.length, Math.ceil(t1 / dt));
    const fSlice = fhr.slice(i0, i1);
    const f = finite(fSlice);
    const m = finite(mhr.slice(i0, i1));
    const c = finite(tocoPct.slice(i0, i1));
    const baseline = median(f);
    const low = percentile(f, 10);
    const high = percentile(f, 90);
    out.push({
      t0,
      t1,
      baseline: roundOrNull(baseline, 0),
      low: roundOrNull(low, 0),
      high: roundOrNull(high, 0),
      band: roundOrNull(low == null || high == null ? null : high - low, 0),
      rises: baseline == null ? 0 : episodes(fSlice, dt, (v) => v >= baseline + 15, 15),
      falls: baseline == null ? 0 : episodes(fSlice, dt, (v) => v <= baseline - 15, 15),
      mhr: roundOrNull(median(m), 0),
      toco: roundOrNull(median(c), 1),
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

function csvCell(v: CtgSample | undefined): string {
  return isSample(v) ? String(v) : "";
}

export function toCsv(session: CtgSession): string {
  const factor = session.tocoKpaPerPct;
  const withKpa = factor != null && Number.isFinite(factor);
  const lines = [withKpa ? "t_sec,clock,fhr_bpm,mhr_bpm,toco_pct,toco_kpa" : "t_sec,clock,fhr_bpm,mhr_bpm,toco_pct"];
  for (let i = 0; i < session.fhr.length; i++) {
    const t = i * session.dt;
    const cells = [
      String(t),
      clockOf(session.startedAt, t),
      csvCell(session.fhr[i]),
      csvCell(session.mhr[i]),
      csvCell(session.tocoPct[i]),
    ];
    if (withKpa) {
      const kpa = tocoKpa(session.tocoPct[i], factor);
      cells.push(kpa == null ? "" : String(kpa));
    }
    lines.push(cells.join(","));
  }
  return lines.join("\n");
}
