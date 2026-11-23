import type { CtgSample } from "@/lib/ctg/types";

/**
 * Pure CTG metrics for the analysis view.
 *
 * Paper speed is 1 cm/min and the exported series is 1 Hz, so short-term
 * variability in milliseconds, LF/HF spectra, sample entropy and PRSA are not
 * computed. See `SKIPPED_METHODS_NOTE`.
 *
 * `gzipRatio` is not gzip. `CompressionStream` is asynchronous and `zlib` is
 * not available in the browser bundle, so the ratio is the length of a fixed
 * LZ77 code divided by the raw length (window 64, literal 9 bits, match 11
 * bits, match length 3–18). A lower ratio means a more repetitive, rigid
 * series. Null samples are omitted. Each kept FHR value is rounded, clamped
 * to 50–210 bpm and stored as `value - 50`.
 */

export const SKIPPED_METHODS_NOTE =
  "Nicht berechnet: STV und Dawes-Redman in Millisekunden, das LF/HF-Spektrum, Sample-Entropy und PRSA. Die Abtastung dieser Papieraufzeichnung ist 1 Hz bei 1 cm/min; diese Verfahren brauchen eine feinere Auflösung. Die Ansicht ist keine Diagnose und kein Medizinprodukt.";

/** Scores of 0 and 100 for the six radar axes. */
export const RADAR = {
  driftBpm: 30,
  iqrBpm: 40,
  gzipLow: 0.1,
  gzipHigh: 1,
  aucBpmSec: 12000,
  recoveryBpmPerSec: 1,
  corr: 0.6,
} as const;

const LZ_WINDOW = 64;
const LZ_MAX_LEN = 18;

export type DescriptiveStats = {
  n: number;
  nullFraction: number | null;
  firstValidIndex: number | null;
  lastValidIndex: number | null;
  median: number | null;
  trimmedMean10: number | null;
  mean: number | null;
  std: number | null;
  cv: number | null;
  iqr: number | null;
  p10: number | null;
  p90: number | null;
  band: number | null;
  skewness: number | null;
  excessKurtosis: number | null;
};

export type VariabilityClass = "absent" | "minimal" | "moderate" | "marked";

export type Variability = {
  amplitude: number | null;
  baseline: number | null;
  klass: VariabilityClass | null;
};

export type RadarScores = {
  baselineStabilitaet: number | null;
  variabilitaet: number | null;
  komplexitaet: number | null;
  dezelerationsLast: number | null;
  erholungsrate: number | null;
  kopplung: number | null;
};

export type LagCorr = {
  lags: number[];
  r: (number | null)[];
  bestLag: number | null;
  bestR: number | null;
};

export function isSample(v: CtgSample | undefined): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function finiteValues(values: readonly CtgSample[]): number[] {
  const out: number[] = [];
  for (const v of values) {
    if (isSample(v)) out.push(v);
  }
  return out;
}

export function median(values: readonly number[]): number | null {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Linear quantile on the sorted sample. `p` is 0–100. */
export function quantile(sorted: readonly number[], p: number): number | null {
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

export function descriptiveStats(values: readonly CtgSample[], indexOffset = 0): DescriptiveStats {
  const empty: DescriptiveStats = {
    n: 0,
    nullFraction: values.length ? 0 : null,
    firstValidIndex: null,
    lastValidIndex: null,
    median: null,
    trimmedMean10: null,
    mean: null,
    std: null,
    cv: null,
    iqr: null,
    p10: null,
    p90: null,
    band: null,
    skewness: null,
    excessKurtosis: null,
  };
  if (!values.length) return empty;

  let nulls = 0;
  let first: number | null = null;
  let last: number | null = null;
  const xs: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!isSample(v)) {
      nulls += 1;
      continue;
    }
    if (first == null) first = indexOffset + i;
    last = indexOffset + i;
    xs.push(v);
  }
  const n = xs.length;
  const nullFraction = nulls / values.length;
  if (!n) {
    return { ...empty, nullFraction, n: 0 };
  }

  const sorted = [...xs].sort((a, b) => a - b);
  let sum = 0;
  for (const v of xs) sum += v;
  const mean = sum / n;
  const med = median(sorted);
  const k = Math.floor(n * 0.1);
  let trimmedSum = 0;
  for (let i = k; i < n - k; i++) trimmedSum += sorted[i];
  const trimmedMean10 = trimmedSum / (n - 2 * k);

  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  if (n >= 2) {
    for (const v of xs) {
      const d = v - mean;
      const d2 = d * d;
      m2 += d2;
      m3 += d2 * d;
      m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
  }
  const std = n >= 2 ? Math.sqrt(m2) : null;
  const cv = std != null && Math.abs(mean) > 1e-12 ? std / Math.abs(mean) : null;
  const p25 = quantile(sorted, 25);
  const p75 = quantile(sorted, 75);
  const p10 = quantile(sorted, 10);
  const p90 = quantile(sorted, 90);
  const skewness = std != null && std > 1e-12 && n >= 3 ? m3 / std ** 3 : null;
  const excessKurtosis = std != null && std > 1e-12 && n >= 4 ? m4 / m2 ** 2 - 3 : null;

  return {
    n,
    nullFraction,
    firstValidIndex: first,
    lastValidIndex: last,
    median: med,
    trimmedMean10,
    mean,
    std,
    cv,
    iqr: p25 == null || p75 == null ? null : p75 - p25,
    p10,
    p90,
    band: p10 == null || p90 == null ? null : p90 - p10,
    skewness,
    excessKurtosis,
  };
}

/** Peak-to-trough of the valid FHR samples, then the NICHD amplitude bands. */
export function variability(values: readonly CtgSample[]): Variability {
  const xs = finiteValues(values);
  if (!xs.length) return { amplitude: null, baseline: null, klass: null };
  let lo = xs[0];
  let hi = xs[0];
  for (const v of xs) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const amplitude = hi - lo;
  const med = median(xs);
  const baseline = med == null ? null : Math.round(med);
  let klass: VariabilityClass;
  if (amplitude <= 0) klass = "absent";
  else if (amplitude <= 5) klass = "minimal";
  else if (amplitude <= 25) klass = "moderate";
  else klass = "marked";
  return { amplitude, baseline, klass };
}

export function shannonEntropy(values: readonly number[]): number | null {
  if (!values.length) return null;
  const counts = new Map<number, number>();
  for (const v of values) {
    const bucket = Math.floor(v / 10);
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }
  let h = 0;
  const n = values.length;
  for (const c of counts.values()) {
    const p = c / n;
    h -= p * Math.log2(p);
  }
  return h;
}

/** 50 bpm → 0, 210 bpm → 160. Values outside that span are clamped. */
export function quantizeFhrBytes(values: readonly number[]): Uint8Array {
  const out = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const clamped = Math.min(210, Math.max(50, Math.round(values[i])));
    out[i] = clamped - 50;
  }
  return out;
}

/**
 * LZ77-ish compressed-length / raw-length. See the file header.
 * Overlapping matches are allowed, so a constant run encodes as one literal
 * plus long copies.
 */
export function lz77Ratio(bytes: Uint8Array): number | null {
  const n = bytes.length;
  if (!n) return null;
  let bits = 0;
  let i = 0;
  while (i < n) {
    let bestLen = 0;
    const from = Math.max(0, i - LZ_WINDOW);
    for (let j = from; j < i; j++) {
      if (bytes[j] !== bytes[i]) continue;
      const span = i - j;
      let len = 1;
      const room = Math.min(LZ_MAX_LEN, n - i);
      while (len < room && bytes[j + (len % span)] === bytes[i + len]) len += 1;
      if (len > bestLen) {
        bestLen = len;
        if (bestLen === LZ_MAX_LEN) break;
      }
    }
    if (bestLen >= 3) {
      bits += 11;
      i += bestLen;
    } else {
      bits += 9;
      i += 1;
    }
  }
  return bits / 8 / n;
}

export function gzipRatio(values: readonly number[]): number | null {
  if (!values.length) return null;
  return lz77Ratio(quantizeFhrBytes(values));
}

export function pearson(xs: readonly number[], ys: readonly number[]): number | null {
  const n = xs.length;
  if (n < 2 || ys.length !== n) return null;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  if (dx <= 1e-12 || dy <= 1e-12) return null;
  return num / Math.sqrt(dx * dy);
}

/**
 * Pearson correlation of `a[i]` with `b[i + lag]`, lag in seconds.
 * The reported best lag is the one with the greatest |r|. Ties prefer the
 * lag closest to zero, then the more negative lag.
 */
export function crossCorrelation(
  a: readonly CtgSample[],
  b: readonly CtgSample[],
  dt: number,
  lagFromSec = -60,
  lagToSec = 120,
  stepSec = 5,
  minPairs = 8,
): LagCorr {
  const lags: number[] = [];
  for (let lag = lagFromSec; lag <= lagToSec + 1e-9; lag += stepSec) lags.push(Math.round(lag * 1000) / 1000);
  const r = lags.map((lag) => pairedPearson(a, b, Math.round(lag / dt), minPairs));
  let bestLag: number | null = null;
  let bestR: number | null = null;
  for (let i = 0; i < lags.length; i++) {
    const value = r[i];
    if (value == null) continue;
    if (bestR == null || betterLag(lags[i], value, bestLag ?? 0, bestR)) {
      bestLag = lags[i];
      bestR = value;
    }
  }
  return { lags, r, bestLag, bestR };
}

function betterLag(lag: number, r: number, bestLag: number, bestR: number): boolean {
  const ar = Math.abs(r);
  const br = Math.abs(bestR);
  if (ar > br + 1e-9) return true;
  if (Math.abs(ar - br) > 1e-9) return false;
  if (Math.abs(lag) < Math.abs(bestLag) - 1e-9) return true;
  if (Math.abs(Math.abs(lag) - Math.abs(bestLag)) <= 1e-9 && lag < bestLag) return true;
  return false;
}

function pairedPearson(
  a: readonly CtgSample[],
  b: readonly CtgSample[],
  lagSamples: number,
  minPairs: number,
): number | null {
  const n = Math.min(a.length, b.length);
  const xs: number[] = [];
  const ys: number[] = [];
  const iStart = Math.max(0, -lagSamples);
  const iEnd = Math.min(n, n - lagSamples);
  for (let i = iStart; i < iEnd; i++) {
    const j = i + lagSamples;
    const x = a[i];
    const y = b[j];
    if (!isSample(x) || !isSample(y)) continue;
    xs.push(x);
    ys.push(y);
  }
  if (xs.length < minPairs) return null;
  return pearson(xs, ys);
}

/** Pearson autocorrelation of the valid pairs at lags of 1..maxLagSec seconds. */
export function autocorrelation(
  values: readonly CtgSample[],
  dt: number,
  maxLagSec = 60,
  minPairs = 8,
): { lag: number; r: number | null }[] {
  const steps = Math.max(0, Math.round(maxLagSec / dt));
  const out: { lag: number; r: number | null }[] = [];
  for (let k = 1; k <= steps; k++) {
    out.push({ lag: k * dt, r: pairedPearson(values, values, k, minPairs) });
  }
  return out;
}

/**
 * High periodicity heuristic: the strongest ACF between 15 and 40 s exceeds
 * 0.35 while the FHR IQR stays under 8 bpm.
 */
export function periodicHint(acf: readonly { lag: number; r: number | null }[], iqr: number | null): boolean {
  if (iqr == null || iqr >= 8) return false;
  let max = -Infinity;
  let any = false;
  for (const point of acf) {
    if (point.lag < 15 || point.lag > 40 || point.r == null) continue;
    any = true;
    if (point.r > max) max = point.r;
  }
  return any && max > 0.35;
}

export function meanAbsSlope(values: readonly CtgSample[], dt: number): number | null {
  if (!(dt > 0)) return null;
  let sum = 0;
  let n = 0;
  for (let i = 1; i < values.length; i++) {
    const left = values[i - 1];
    const right = values[i];
    if (!isSample(left) || !isSample(right)) continue;
    sum += Math.abs(right - left) / dt;
    n += 1;
  }
  return n ? sum / n : null;
}

/**
 * |max − min| of rolling medians. Short series that fit in one sub-window
 * report a drift of 0 when they have any valid sample.
 */
export function rollingMedianDrift(
  values: readonly CtgSample[],
  dt: number,
  winSec = 60,
  strideSec = 60,
): number | null {
  if (!(dt > 0) || !values.length) return null;
  const win = Math.max(1, Math.round(winSec / dt));
  const stride = Math.max(1, Math.round(strideSec / dt));
  if (values.length < win) {
    return median(finiteValues(values)) == null ? null : 0;
  }
  const medians: number[] = [];
  for (let i = 0; i + win <= values.length; i += stride) {
    const m = median(finiteValues(values.slice(i, i + win)));
    if (m != null) medians.push(m);
  }
  if (medians.length < 2) return medians.length ? 0 : null;
  let lo = medians[0];
  let hi = medians[0];
  for (const m of medians) {
    if (m < lo) lo = m;
    if (m > hi) hi = m;
  }
  return hi - lo;
}

export function scaleClamp(value: number, zeroAt: number, fullAt: number): number {
  if (fullAt === zeroAt) return 0;
  const t = ((value - zeroAt) / (fullAt - zeroAt)) * 100;
  return Math.min(100, Math.max(0, t));
}

export function radarScores(input: {
  driftBpm: number | null;
  iqr: number | null;
  gzipRatio: number | null;
  totalAuc: number | null;
  meanRecovery: number | null;
  bestAbsR: number | null;
  decelCount: number;
}): RadarScores {
  return {
    baselineStabilitaet:
      input.driftBpm == null ? null : scaleClamp(input.driftBpm, RADAR.driftBpm, 0),
    variabilitaet: input.iqr == null ? null : scaleClamp(input.iqr, 0, RADAR.iqrBpm),
    komplexitaet:
      input.gzipRatio == null ? null : scaleClamp(input.gzipRatio, RADAR.gzipLow, RADAR.gzipHigh),
    dezelerationsLast: input.totalAuc == null ? null : scaleClamp(input.totalAuc, 0, RADAR.aucBpmSec),
    erholungsrate:
      input.decelCount === 0 || input.meanRecovery == null
        ? null
        : scaleClamp(input.meanRecovery, 0, RADAR.recoveryBpmPerSec),
    kopplung: input.bestAbsR == null ? null : scaleClamp(input.bestAbsR, 0, RADAR.corr),
  };
}
