import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { formatSample, toCsv, windowStats } from "./analyze.ts";
import type { CtgSession } from "./types.ts";

function session(partial: Partial<CtgSession> = {}): CtgSession {
  return {
    id: "t",
    title: "t",
    startedAt: "2026-11-22T11:10:12",
    durationSec: 3,
    dt: 1,
    device: "Philips M2702A",
    paper: "M1911A",
    speedCmMin: 1,
    fhrScale: { min: 50, max: 210, major: 20, label: "bpm" },
    tocoScale: { min: 0, max: 100, major: 10, label: "Skala" },
    provenance: { trace: "t", scans: "s", note: "n" },
    fhr: [100, null, 120],
    mhr: [null, 80, 81],
    toco: [1.2, null, 1.5],
    tocoPct: [20, null, 25],
    panorama: {
      file: "panorama.webp",
      widthPx: 10,
      heightPx: 10,
      pxPerCm: 100,
      fhr210YPx: 1,
      fhr50YPx: 5,
      toco100YPx: 6,
      toco0YPx: 9,
      fhrHeightCm: 8,
      separatorHeightCm: 0.675,
      tocoHeightCm: 4,
      fhrMinorCm: 0.25,
      tocoMinorCm: 0.4,
      verticalMinorCm: 0.25,
      fhrTopBpm: 210,
      fhrBotBpm: 50,
      tocoTop: 100,
      tocoBot: 0,
    },
    events: [],
    scans: [],
    ...partial,
  };
}

describe("null samples", () => {
  it("ignores nulls in window statistics and leaves an empty window empty", () => {
    const [window] = windowStats(session(), 10);
    assert.equal(window.baseline, 110);
    assert.equal(window.low, 100);
    assert.equal(window.high, 120);
    assert.equal(window.mhr, 81);
    assert.equal(window.toco, 22.5);
    assert.equal(window.falls, 0);
    assert.equal(window.rises, 0);

    const empty = windowStats(
      session({ fhr: [null, null, null], mhr: [null, null, null], tocoPct: [null, null, null] }),
      10,
    )[0];
    assert.equal(empty.baseline, null);
    assert.equal(empty.low, null);
    assert.equal(empty.high, null);
    assert.equal(empty.band, null);
    assert.equal(empty.mhr, null);
    assert.equal(empty.toco, null);
    assert.equal(empty.rises, 0);
    assert.equal(empty.falls, 0);
  });

  it("does not treat a null as zero when counting rises and falls", () => {
    const fhr = Array.from({ length: 40 }, (_, i) => (i >= 10 && i < 30 ? null : 160));
    const [window] = windowStats(session({ fhr, mhr: fhr, tocoPct: fhr, durationSec: 40 }), 60);
    assert.equal(window.baseline, 160);
    assert.equal(window.falls, 0);
    assert.equal(window.rises, 0);
  });

  it("writes empty CSV cells and adds kPa only when the factor is present", () => {
    const csv = toCsv(session());
    const lines = csv.split("\n");
    assert.equal(lines[0], "t_sec,clock,fhr_bpm,mhr_bpm,toco_pct");
    assert.equal(lines[1], "0,11:10:12,100,,20");
    assert.equal(lines[2], "1,11:10:13,,80,");
    assert.equal(lines[3], "2,11:10:14,120,81,25");

    const withFactor = toCsv(session({ tocoKpaPerPct: 0.12 })).split("\n");
    assert.equal(withFactor[0], "t_sec,clock,fhr_bpm,mhr_bpm,toco_pct,toco_kpa");
    assert.equal(withFactor[1], "0,11:10:12,100,,20,2.4");
    assert.equal(withFactor[2], "1,11:10:13,,80,,");
    assert.equal(formatSample(null, 1), "–");
    assert.equal(formatSample(undefined, 0), "–");
  });
});
