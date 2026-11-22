export type CtgEvent = {
  t: number;
  kind: "mark" | "puls" | "kineto";
  text: string;
  value?: number;
  extra?: number;
};

export type CtgScan = {
  file: string;
  t0: number;
  t1: number;
  trust: "hoch" | "mittel" | "niedrig";
};

/** One sample per second. Null where the pipeline found no trace on the paper. */
export type CtgSample = number | null;

/**
 * Placement of the stitched panorama and the paper geometry it was rectified to.
 * All distances are in centimetres on the physical paper; pixel rows are in the
 * image file the viewer loads.
 */
export type CtgPanorama = {
  file: string;
  widthPx: number;
  heightPx: number;
  pxPerCm: number;
  fhr210YPx: number;
  fhr50YPx: number;
  toco100YPx: number;
  toco0YPx: number;
  fhrHeightCm: number;
  separatorHeightCm: number;
  tocoHeightCm: number;
  fhrMinorCm: number;
  tocoMinorCm: number;
  verticalMinorCm: number;
  fhrTopBpm: number;
  fhrBotBpm: number;
  tocoTop: number;
  tocoBot: number;
};

export type CtgSession = {
  id: string;
  title: string;
  startedAt: string;
  durationSec: number;
  dt: number;
  device: string;
  paper: string;
  speedCmMin: number;
  fhrScale: { min: number; max: number; major: number; label: string };
  tocoScale: { min: number; max: number; major: number; label: string };
  provenance: { trace: string; scans: string; note: string };
  fhr: CtgSample[];
  mhr: CtgSample[];
  toco: CtgSample[];
  /** Toco on the paper's 0 to 100 scale. This is the series the strip draws. */
  tocoPct: CtgSample[];
  /**
   * Optional kPa per toco-scale unit. Absent while that factor is unverified.
   * kPa is only shown as a secondary label when this field is present.
   */
  tocoKpaPerPct?: number;
  panorama: CtgPanorama;
  events: CtgEvent[];
  scans: CtgScan[];
};
