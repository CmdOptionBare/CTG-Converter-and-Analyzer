import type { CtgSample } from "@/lib/ctg/types";
import { isSample, median } from "./metrics.ts";

export type Deceleration = {
  t0: number;
  t1: number;
  troughT: number;
  troughFhr: number;
  depth: number;
  durationSec: number;
  /** ∫(baseline − fhr) dt over the run, in bpm·s. */
  auc: number;
  /** (fhr at trough − fhr at t0) / (trough − t0). Null when the trough is the first sample. */
  descentSlope: number | null;
  /** Rise from the trough to the latest valid sample inside the next 30 s, bpm/s. */
  recoverySlope: number | null;
  tocoPeakT: number | null;
  tocoPeak: number | null;
  /** trough time − toco peak time. Positive when the toco peak leads the trough. */
  lagSec: number | null;
  gzipRatio: number | null;
};

export type TocoEvent = {
  t0: number;
  t1: number;
  peakT: number;
  peak: number;
  height: number;
  durationSec: number;
};

const DECEL_DROP_BPM = 15;
const DECEL_MIN_SEC = 15;
const RECOVERY_SEC = 30;
const TOCO_LOOKBACK_SEC = 60;
const TOCO_LOOKAHEAD_SEC = 90;
const TOCO_RISE = 8;
const TOCO_MIN_SEC = 20;

/**
 * Contiguous runs of at least 15 s with FHR ≤ baseline − 15 bpm.
 * A null sample ends the run. `baseline` is the median of the valid FHR
 * on the recording the caller cares about (the whole trace for this app).
 */
export function detectDecelerations(
  fhr: readonly CtgSample[],
  toco: readonly CtgSample[],
  dt: number,
  baseline: number,
): Deceleration[] {
  if (!(dt > 0) || !Number.isFinite(baseline)) return [];
  const threshold = baseline - DECEL_DROP_BPM;
  const out: Deceleration[] = [];
  let i = 0;
  while (i < fhr.length) {
    const sample = fhr[i];
    if (!isSample(sample) || sample > threshold) {
      i += 1;
      continue;
    }
    let j = i;
    while (j < fhr.length && isSample(fhr[j]) && (fhr[j] as number) <= threshold) j += 1;
    const count = j - i;
    const durationSec = count * dt;
    if (durationSec + 1e-9 >= DECEL_MIN_SEC) {
      out.push(describeRun(fhr, toco, dt, baseline, i, j));
    }
    i = j;
  }
  return out;
}

function describeRun(
  fhr: readonly CtgSample[],
  toco: readonly CtgSample[],
  dt: number,
  baseline: number,
  i0: number,
  i1: number,
): Deceleration {
  let troughIndex = i0;
  let troughFhr = fhr[i0] as number;
  let auc = 0;
  for (let i = i0; i < i1; i++) {
    const v = fhr[i] as number;
    auc += (baseline - v) * dt;
    if (v < troughFhr) {
      troughFhr = v;
      troughIndex = i;
    }
  }
  const t0 = i0 * dt;
  const t1 = i1 * dt;
  const troughT = troughIndex * dt;
  const startFhr = fhr[i0] as number;
  const descentSlope = troughIndex === i0 ? null : (troughFhr - startFhr) / (troughT - t0);
  const recoverySlope = recovery(fhr, dt, troughIndex, troughFhr);
  const tocoPeak = nearestTocoPeak(toco, dt, troughT);
  return {
    t0,
    t1,
    troughT,
    troughFhr,
    depth: baseline - troughFhr,
    durationSec: t1 - t0,
    auc,
    descentSlope,
    recoverySlope,
    tocoPeakT: tocoPeak?.t ?? null,
    tocoPeak: tocoPeak?.value ?? null,
    lagSec: tocoPeak ? troughT - tocoPeak.t : null,
    gzipRatio: null,
  };
}

function recovery(fhr: readonly CtgSample[], dt: number, troughIndex: number, troughFhr: number): number | null {
  const horizon = troughIndex + Math.round(RECOVERY_SEC / dt);
  let bestIndex = -1;
  const last = Math.min(fhr.length - 1, horizon);
  for (let i = troughIndex + 1; i <= last; i++) {
    if (isSample(fhr[i])) bestIndex = i;
  }
  if (bestIndex < 0) return null;
  const elapsed = (bestIndex - troughIndex) * dt;
  if (!(elapsed > 0)) return null;
  return ((fhr[bestIndex] as number) - troughFhr) / elapsed;
}

function nearestTocoPeak(
  toco: readonly CtgSample[],
  dt: number,
  troughT: number,
): { t: number; value: number } | null {
  const tLo = troughT - TOCO_LOOKBACK_SEC;
  const tHi = troughT + TOCO_LOOKAHEAD_SEC;
  const n = toco.length;
  const i0 = Math.max(0, Math.ceil(tLo / dt));
  const i1 = Math.min(n - 1, Math.floor(tHi / dt));
  const windowValues: number[] = [];
  for (let i = i0; i <= i1; i++) {
    if (isSample(toco[i])) windowValues.push(toco[i] as number);
  }
  const med = median(windowValues);
  if (med == null) return null;
  // A flat trace has an argmax, but not a peak. Require a local rise.
  const level = med + 2;
  let best: { t: number; value: number } | null = null;
  let bestDist = Infinity;
  for (let i = i0; i <= i1; i++) {
    const v = toco[i];
    if (!isSample(v) || v < level || !isLocalPeak(toco, i)) continue;
    const t = i * dt;
    const dist = Math.abs(t - troughT);
    if (
      best == null ||
      dist < bestDist - 1e-9 ||
      (Math.abs(dist - bestDist) <= 1e-9 && (v > best.value + 1e-9 || (Math.abs(v - best.value) <= 1e-9 && t < best.t)))
    ) {
      best = { t, value: v };
      bestDist = dist;
    }
  }
  return best;
}

function isLocalPeak(toco: readonly CtgSample[], index: number): boolean {
  const v = toco[index];
  if (!isSample(v)) return false;
  const left = index > 0 ? toco[index - 1] : null;
  const right = index + 1 < toco.length ? toco[index + 1] : null;
  const aboveLeft = !isSample(left) || v > (left as number);
  const atLeastRight = !isSample(right) || v >= (right as number);
  return aboveLeft && atLeastRight;
}

/**
 * A contraction is a contiguous run of at least 20 s at least 8 units above
 * the median of the valid toco samples. Nulls break the run.
 */
export function detectTocoEvents(toco: readonly CtgSample[], dt: number): TocoEvent[] {
  if (!(dt > 0)) return [];
  const xs: number[] = [];
  for (const v of toco) {
    if (isSample(v)) xs.push(v);
  }
  const med = median(xs);
  if (med == null) return [];
  const level = med + TOCO_RISE;
  const out: TocoEvent[] = [];
  let i = 0;
  while (i < toco.length) {
    const sample = toco[i];
    if (!isSample(sample) || sample < level) {
      i += 1;
      continue;
    }
    let j = i;
    while (j < toco.length && isSample(toco[j]) && (toco[j] as number) >= level) j += 1;
    const durationSec = (j - i) * dt;
    if (durationSec + 1e-9 >= TOCO_MIN_SEC) {
      let peak = toco[i] as number;
      let peakIndex = i;
      for (let k = i; k < j; k++) {
        const v = toco[k] as number;
        if (v > peak) {
          peak = v;
          peakIndex = k;
        }
      }
      out.push({
        t0: i * dt,
        t1: j * dt,
        peakT: peakIndex * dt,
        peak,
        height: peak - med,
        durationSec,
      });
    }
    i = j;
  }
  return out;
}
