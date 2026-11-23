import type { CtgSample, CtgSession } from "@/lib/ctg/types";
import { detectDecelerations, detectTocoEvents, type Deceleration, type TocoEvent } from "./detectors.ts";
import {
  autocorrelation,
  crossCorrelation,
  descriptiveStats,
  finiteValues,
  gzipRatio,
  meanAbsSlope,
  median,
  periodicHint,
  radarScores,
  rollingMedianDrift,
  shannonEntropy,
  variability,
  type DescriptiveStats,
  type LagCorr,
  type RadarScores,
  type Variability,
} from "./metrics.ts";

export type TimeRange = { t0: number; t1: number };

export type ScopeSummary = {
  t0: number;
  t1: number;
  fhr: DescriptiveStats;
  mhr: DescriptiveStats;
  toco: DescriptiveStats;
  variability: Variability;
  entropy: number | null;
  gzipRatio: number | null;
  meanAbsSlope: number | null;
  decelerations: Deceleration[];
  tocoEvents: TocoEvent[];
  totalAuc: number;
  meanRecovery: number | null;
  cross: LagCorr;
  fhrMhrR: number | null;
  acf: { lag: number; r: number | null }[];
  periodic: boolean;
  driftBpm: number | null;
  radar: RadarScores;
};

export type CorrSurface = {
  winSec: number;
  strideSec: number;
  t0: number[];
  lags: number[];
  r: (number | null)[][];
  bestNegLag: (number | null)[];
  bestNegR: (number | null)[];
};

export type SessionAnalysis = {
  dt: number;
  durationSec: number;
  globalBaseline: number | null;
  global: ScopeSummary;
  decels: Deceleration[];
  tocoEvents: TocoEvent[];
  surface: CorrSurface;
  gaps: TimeRange[];
};

export function rollingRanges(durationSec: number, winSec: number, strideSec: number): TimeRange[] {
  if (!(durationSec > 0) || !(winSec > 0) || !(strideSec > 0)) return [];
  const out: TimeRange[] = [];
  for (let t0 = 0; t0 + winSec <= durationSec + 1e-6; t0 += strideSec) {
    out.push({ t0, t1: Math.min(durationSec, t0 + winSec) });
  }
  return out;
}

export function sliceRange<T>(values: readonly T[], dt: number, t0: number, t1: number): { values: T[]; i0: number } {
  const n = values.length;
  const i0 = Math.max(0, Math.min(n, Math.floor(t0 / dt)));
  const i1 = Math.max(i0, Math.min(n, Math.ceil(t1 / dt)));
  return { values: values.slice(i0, i1), i0 };
}

export function eventWindow(t0: number, t1: number, durationSec: number, padBefore = 60, padAfter = 60): TimeRange {
  return {
    t0: Math.max(0, t0 - padBefore),
    t1: Math.min(durationSec, t1 + padAfter),
  };
}

/** Null runs of at least `minSec`, including the leading and trailing gaps. */
export function nullGaps(values: readonly CtgSample[], dt: number, minSec = 5): TimeRange[] {
  const out: TimeRange[] = [];
  let i = 0;
  while (i < values.length) {
    if (values[i] != null && Number.isFinite(values[i] as number)) {
      i += 1;
      continue;
    }
    let j = i;
    while (j < values.length && (values[j] == null || !Number.isFinite(values[j] as number))) j += 1;
    const duration = (j - i) * dt;
    if (duration + 1e-9 >= minSec) out.push({ t0: i * dt, t1: j * dt });
    i = j;
  }
  return out;
}

export function analyzeSession(session: CtgSession): SessionAnalysis {
  const dt = session.dt || 1;
  const baseline = median(finiteValues(session.fhr));
  const detected = baseline == null ? [] : detectDecelerations(session.fhr, session.tocoPct, dt, baseline);
  const decels = detected.map((event) => {
    const around = sliceRange(session.fhr, dt, event.troughT - 30, event.troughT + 30);
    return { ...event, gzipRatio: gzipRatio(finiteValues(around.values)) };
  });
  const tocoEvents = detectTocoEvents(session.tocoPct, dt);
  const global = summarizeScope(session, 0, session.durationSec, baseline, decels, tocoEvents);
  return {
    dt,
    durationSec: session.durationSec,
    globalBaseline: baseline,
    global,
    decels,
    tocoEvents,
    surface: correlationSurface(session.fhr, session.tocoPct, dt, session.durationSec, 600, 120),
    gaps: nullGaps(session.fhr, dt, 5),
  };
}

export function summarizeScope(
  session: CtgSession,
  t0: number,
  t1: number,
  globalBaseline: number | null,
  decels: readonly Deceleration[],
  tocoEvents: readonly TocoEvent[],
): ScopeSummary {
  const dt = session.dt || 1;
  const fhr = sliceRange(session.fhr, dt, t0, t1);
  const mhr = sliceRange(session.mhr, dt, t0, t1);
  const toco = sliceRange(session.tocoPct, dt, t0, t1);
  const fhrStats = descriptiveStats(fhr.values, fhr.i0);
  const kept = finiteValues(fhr.values);
  const inScopeDecels = decels.filter((event) => event.troughT >= t0 && event.troughT < t1);
  const inScopeToco = tocoEvents.filter((event) => event.peakT >= t0 && event.peakT < t1);
  const totalAuc = inScopeDecels.reduce((sum, event) => sum + event.auc, 0);
  const recoveries = inScopeDecels.map((event) => event.recoverySlope).filter((v): v is number => v != null);
  const meanRecovery = recoveries.length ? recoveries.reduce((s, v) => s + v, 0) / recoveries.length : null;
  const cross = crossCorrelation(fhr.values, toco.values, dt);
  const fhrMhr = crossCorrelation(fhr.values, mhr.values, dt, 0, 0, 1);
  const acf = autocorrelation(fhr.values, dt, 60);
  const scopeSec = fhr.values.length * dt;
  const sub = scopeSec >= 180 ? 60 : Math.max(dt, scopeSec / 3);
  const driftBpm = rollingMedianDrift(fhr.values, dt, sub, sub);
  const ratio = gzipRatio(kept);
  return {
    t0,
    t1,
    fhr: fhrStats,
    mhr: descriptiveStats(mhr.values, mhr.i0),
    toco: descriptiveStats(toco.values, toco.i0),
    variability: variability(fhr.values),
    entropy: shannonEntropy(kept),
    gzipRatio: ratio,
    meanAbsSlope: meanAbsSlope(fhr.values, dt),
    decelerations: inScopeDecels,
    tocoEvents: inScopeToco,
    totalAuc,
    meanRecovery,
    cross,
    fhrMhrR: fhrMhr.bestR,
    acf,
    periodic: periodicHint(acf, fhrStats.iqr),
    driftBpm,
    radar: radarScores({
      driftBpm,
      iqr: fhrStats.iqr,
      gzipRatio: ratio,
      totalAuc,
      meanRecovery,
      bestAbsR: cross.bestR == null ? null : Math.abs(cross.bestR),
      decelCount: inScopeDecels.length,
    }),
  };
}

export function correlationSurface(
  fhr: readonly CtgSample[],
  toco: readonly CtgSample[],
  dt: number,
  durationSec: number,
  winSec = 600,
  strideSec = 120,
): CorrSurface {
  const ranges = rollingRanges(durationSec, winSec, strideSec);
  const rows: (number | null)[][] = [];
  const bestNegLag: (number | null)[] = [];
  const bestNegR: (number | null)[] = [];
  let lags: number[] = [];
  for (const range of ranges) {
    const a = sliceRange(fhr, dt, range.t0, range.t1).values;
    const b = sliceRange(toco, dt, range.t0, range.t1).values;
    const corr = crossCorrelation(a, b, dt);
    if (!lags.length) lags = corr.lags;
    rows.push(corr.r);
    let lag: number | null = null;
    let value: number | null = null;
    for (let i = 0; i < corr.lags.length; i++) {
      if (corr.lags[i] > 0 || corr.r[i] == null) continue;
      const r = corr.r[i] as number;
      if (value == null || Math.abs(r) > Math.abs(value) + 1e-12) {
        lag = corr.lags[i];
        value = r;
      }
    }
    bestNegLag.push(lag);
    bestNegR.push(value);
  }
  if (!lags.length) {
    const probe = crossCorrelation([], [], dt);
    lags = probe.lags;
  }
  return { winSec, strideSec, t0: ranges.map((range) => range.t0), lags, r: rows, bestNegLag, bestNegR };
}
