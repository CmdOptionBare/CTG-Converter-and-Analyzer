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
  fhr: number[];
  mhr: number[];
  toco: number[];
  events: CtgEvent[];
  scans: CtgScan[];
};
