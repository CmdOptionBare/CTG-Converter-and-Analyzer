import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { detectDecelerations, detectTocoEvents } from "./detectors.ts";
import {
  autocorrelation,
  crossCorrelation,
  descriptiveStats,
  gzipRatio,
  meanAbsSlope,
  periodicHint,
  quantizeFhrBytes,
  radarScores,
  shannonEntropy,
  variability,
} from "./metrics.ts";
import { nullGaps, rollingRanges } from "./windows.ts";

describe("descriptive stats", () => {
  it("matches a known series and skips nulls", () => {
    const values = [null, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, null];
    const stats = descriptiveStats(values);
    assert.equal(stats.n, 10);
    assert.equal(stats.nullFraction, 2 / 12);
    assert.equal(stats.firstValidIndex, 1);
    assert.equal(stats.lastValidIndex, 10);
    assert.equal(stats.median, 5.5);
    assert.equal(stats.trimmedMean10, 5.5);
    assert.equal(stats.mean, 5.5);
    assert.ok(Math.abs((stats.std ?? 0) - Math.sqrt(8.25)) < 1e-9);
    assert.equal(stats.iqr, 4.5);
    assert.ok(Math.abs((stats.p10 ?? 0) - 1.9) < 1e-9);
    assert.ok(Math.abs((stats.p90 ?? 0) - 9.1) < 1e-9);
    assert.ok(Math.abs((stats.band ?? 0) - 7.2) < 1e-9);
    assert.ok(Math.abs(stats.skewness ?? 1) < 1e-9);
    assert.ok(stats.excessKurtosis != null && stats.excessKurtosis < 0);

    const skewed = descriptiveStats([1, 1, 1, 1, 5]);
    assert.equal(skewed.median, 1);
    assert.ok(Math.abs((skewed.mean ?? 0) - 1.8) < 1e-12);
    assert.ok(Math.abs((skewed.std ?? 0) - 1.6) < 1e-12);
    assert.ok(Math.abs((skewed.skewness ?? 0) - 1.5) < 1e-9);
    assert.ok(Math.abs((skewed.excessKurtosis ?? 0) - 0.25) < 1e-9);
  });

  it("trims ten percent from each tail", () => {
    const stats = descriptiveStats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
    assert.equal(stats.trimmedMean10, 5.5);
    assert.equal(stats.median, 5.5);
  });

  it("leaves an empty or all-null series empty", () => {
    const empty = descriptiveStats([]);
    assert.equal(empty.n, 0);
    assert.equal(empty.median, null);
    assert.equal(empty.nullFraction, null);
    const blanks = descriptiveStats([null, null]);
    assert.equal(blanks.n, 0);
    assert.equal(blanks.nullFraction, 1);
    assert.equal(blanks.firstValidIndex, null);
    assert.equal(blanks.std, null);
    assert.equal(blanks.iqr, null);
  });
});

describe("variability", () => {
  it("classifies peak-to-trough amplitude and rounds the median baseline", () => {
    assert.equal(variability([100, 100, 100]).klass, "absent");
    assert.equal(variability([100, 105]).klass, "minimal");
    assert.equal(variability([100, 105]).amplitude, 5);
    const moderate = variability([100, 110, null, 105]);
    assert.equal(moderate.amplitude, 10);
    assert.equal(moderate.baseline, 105);
    assert.equal(moderate.klass, "moderate");
    assert.equal(variability([100, 126]).klass, "marked");
    assert.equal(variability([null, null]).klass, null);
  });
});

describe("complexity", () => {
  it("quantizes 50–210 bpm into bytes and gives a constant series a lower ratio", () => {
    const bytes = quantizeFhrBytes([50, 140, 210, 40, 250]);
    assert.deepEqual(Array.from(bytes), [0, 90, 160, 0, 160]);
    const rigid = gzipRatio(Array.from({ length: 240 }, () => 140));
    const busy = gzipRatio(Array.from({ length: 240 }, (_, i) => 50 + ((i * 53) % 160)));
    assert.ok(rigid != null && rigid < 0.25);
    assert.ok(busy != null && rigid != null && rigid < busy * 0.5);
    assert.equal(gzipRatio([]), null);
  });

  it("puts equal 10 bpm buckets at one bit of entropy", () => {
    assert.equal(shannonEntropy([100, 100, 110, 110]), 1);
    assert.equal(shannonEntropy([140, 141, 149]), 0);
    assert.equal(shannonEntropy([]), null);
  });
});

describe("decelerations", () => {
  it("computes AUC, depth and the lag to the nearest toco peak", () => {
    const fhr = Array.from({ length: 80 }, () => 150);
    for (let i = 20; i < 40; i++) fhr[i] = 120;
    const toco = Array.from({ length: 80 }, () => 20);
    toco[10] = 40;
    const [event] = detectDecelerations(fhr, toco, 1, 150);
    assert.equal(event.t0, 20);
    assert.equal(event.t1, 40);
    assert.equal(event.durationSec, 20);
    assert.equal(event.depth, 30);
    assert.equal(event.auc, 600);
    assert.equal(event.troughT, 20);
    assert.equal(event.descentSlope, null);
    assert.equal(event.recoverySlope, 1);
    assert.equal(event.tocoPeakT, 10);
    assert.equal(event.lagSec, 10);
  });

  it("measures descent and a recovery that stops short of 30 s", () => {
    const fhr = Array.from({ length: 60 }, () => 150);
    fhr[10] = 130;
    fhr[11] = 120;
    fhr[12] = 110;
    for (let i = 13; i <= 30; i++) fhr[i] = 110;
    for (let i = 31; i <= 41; i++) fhr[i] = null as unknown as number;
    fhr[42] = 150;
    const [event] = detectDecelerations(fhr, Array.from({ length: 60 }, () => 20), 1, 150);
    assert.equal(event.durationSec, 21);
    assert.equal(event.depth, 40);
    assert.equal(event.auc, 810);
    assert.equal(event.descentSlope, -10);
    assert.ok(Math.abs((event.recoverySlope ?? 0) - 40 / 30) < 1e-12);
  });

  it("does not join runs across a null or keep a run shorter than 15 s", () => {
    const split = Array.from({ length: 40 }, () => 150);
    for (let i = 5; i < 15; i++) split[i] = 120;
    split[15] = null as unknown as number;
    for (let i = 16; i < 26; i++) split[i] = 120;
    assert.equal(detectDecelerations(split, split, 1, 150).length, 0);

    const short = Array.from({ length: 40 }, () => 150);
    for (let i = 10; i < 24; i++) short[i] = 100;
    assert.equal(detectDecelerations(short, short, 1, 150).length, 0);
    for (let i = 10; i < 25; i++) short[i] = 100;
    const [event] = detectDecelerations(short, short, 1, 150);
    assert.equal(event.durationSec, 15);
    assert.equal(event.auc, 750);
  });

  it("picks the closer toco peak when two peaks share a height", () => {
    const fhr = Array.from({ length: 40 }, () => 150);
    for (let i = 10; i < 25; i++) fhr[i] = 100;
    const toco = Array.from({ length: 40 }, () => 10);
    toco[5] = 40;
    toco[15] = 40;
    const [event] = detectDecelerations(fhr, toco, 1, 150);
    assert.equal(event.troughT, 10);
    assert.equal(event.tocoPeakT, 5);
    assert.equal(event.lagSec, 5);
  });
});

describe("toco events", () => {
  it("keeps a rise of at least 8 units lasting at least 20 s", () => {
    const toco = Array.from({ length: 200 }, () => 20);
    for (let i = 40; i < 65; i++) toco[i] = 30;
    const [event] = detectTocoEvents(toco, 1);
    assert.equal(event.durationSec, 25);
    assert.equal(event.peak, 30);
    assert.equal(event.height, 10);
    assert.equal(event.t0, 40);

    const brief = toco.slice();
    for (let i = 40; i < 65; i++) brief[i] = 20;
    for (let i = 40; i < 59; i++) brief[i] = 30;
    assert.equal(detectTocoEvents(brief, 1).length, 0);

    const small = toco.slice();
    for (let i = 40; i < 65; i++) small[i] = 27;
    assert.equal(detectTocoEvents(small, 1).length, 0);
  });
});

describe("cross correlation", () => {
  it("finds the lag of a sine shifted by 10 s, including across nulls", () => {
    const n = 400;
    const fhr = Array.from({ length: n }, (_, t) => 140 + 10 * Math.sin((2 * Math.PI * t) / 60));
    const toco = Array.from({ length: n }, (_, t) => 20 + 5 * Math.sin((2 * Math.PI * (t - 10)) / 60));
    const corr = crossCorrelation(fhr, toco, 1);
    assert.equal(corr.bestLag, 10);
    assert.ok((corr.bestR ?? 0) > 0.99);

    const gappy = fhr.map((v, i) => (i % 7 === 0 ? null : v));
    const gappyCorr = crossCorrelation(gappy, toco, 1);
    assert.equal(gappyCorr.bestLag, 10);
    assert.ok((gappyCorr.bestR ?? 0) > 0.95);
  });

  it("reports a negative r when the series are mirrored", () => {
    const n = 180;
    const fhr = Array.from({ length: n }, (_, t) => 140 + 8 * Math.sin((2 * Math.PI * t) / 40));
    const mirrored = fhr.map((v) => 280 - v);
    const corr = crossCorrelation(fhr, mirrored, 1, -40, 40, 5);
    assert.equal(corr.bestLag, 0);
    assert.ok((corr.bestR ?? 0) < -0.99);
  });

  it("correlates FHR with itself at lag 0 and flags a small periodic swing", () => {
    const flat = crossCorrelation([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], 1, 0, 0, 1);
    assert.equal(flat.bestLag, 0);
    assert.ok((flat.bestR ?? 0) > 0.999);

    const quiet = Array.from({ length: 300 }, (_, t) => 140 + 2 * Math.sin((2 * Math.PI * t) / 30));
    const loud = quiet.map((v) => 140 + (v - 140) * 20);
    const quietAcf = autocorrelation(quiet, 1, 60);
    const loudAcf = autocorrelation(loud, 1, 60);
    assert.equal(periodicHint(quietAcf, descriptiveStats(quiet).iqr), true);
    assert.equal(periodicHint(loudAcf, descriptiveStats(loud).iqr), false);
    const lag30 = quietAcf.find((point) => point.lag === 30);
    assert.ok((lag30?.r ?? 0) > 0.95);
  });
});

describe("slopes, gaps, windows and radar", () => {
  it("skips nulls in the first difference and in gap marks", () => {
    assert.equal(meanAbsSlope([100, 110, null, 130], 1), 10);
    const gaps = nullGaps([null, null, null, null, null, 1, null, null, 2], 1, 5);
    assert.deepEqual(gaps, [{ t0: 0, t1: 5 }]);
  });

  it("steps full windows of 5, 10 and 15 minutes by 2 minutes", () => {
    const ten = rollingRanges(4700, 600, 120);
    assert.equal(ten.length, 35);
    assert.equal(ten[0].t0, 0);
    assert.equal(ten[ten.length - 1].t0, 4080);
    assert.equal(ten[ten.length - 1].t1, 4680);
    assert.equal(rollingRanges(4700, 300, 120).length, 37);
    assert.equal(rollingRanges(4700, 900, 120).length, 32);
  });

  it("clamps the six radar axes", () => {
    const low = radarScores({
      driftBpm: 0,
      iqr: 0,
      gzipRatio: 0.1,
      totalAuc: 0,
      meanRecovery: 0,
      bestAbsR: 0,
      decelCount: 2,
    });
    assert.equal(low.baselineStabilitaet, 100);
    assert.equal(low.variabilitaet, 0);
    assert.equal(low.komplexitaet, 0);
    assert.equal(low.dezelerationsLast, 0);
    assert.equal(low.erholungsrate, 0);
    assert.equal(low.kopplung, 0);

    const high = radarScores({
      driftBpm: 45,
      iqr: 80,
      gzipRatio: 1.2,
      totalAuc: 9000,
      meanRecovery: 2,
      bestAbsR: 0.9,
      decelCount: 1,
    });
    assert.equal(high.baselineStabilitaet, 0);
    assert.equal(high.variabilitaet, 100);
    assert.equal(high.komplexitaet, 100);
    assert.equal(high.dezelerationsLast, 100);
    assert.equal(high.erholungsrate, 100);
    assert.equal(high.kopplung, 100);

    const none = radarScores({
      driftBpm: null,
      iqr: null,
      gzipRatio: null,
      totalAuc: 0,
      meanRecovery: null,
      bestAbsR: null,
      decelCount: 0,
    });
    assert.equal(none.erholungsrate, null);
    assert.equal(none.baselineStabilitaet, null);
  });
});
