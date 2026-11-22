import { useEffect, useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { sessions } from "@/data/sessions";
import type { Deceleration } from "@/lib/ctg-analysis/detectors";
import { clockOf, formatDe, formatSigned } from "@/lib/ctg-analysis/format";
import { SKIPPED_METHODS_NOTE, type VariabilityClass } from "@/lib/ctg-analysis/metrics";
import {
  analyzeSession,
  eventWindow,
  rollingRanges,
  sliceRange,
  summarizeScope,
  type ScopeSummary,
  type TimeRange,
} from "@/lib/ctg-analysis/windows";
import { BarcodePlot } from "./BarcodePlot";
import { ChartFrame } from "./ChartFrame";
import { CrossCorrHeatmap } from "./CrossCorrHeatmap";
import { HorizonChart } from "./HorizonChart";
import { MetricStrip, type MetricItem } from "./MetricStrip";
import { ParallelCoordinates } from "./ParallelCoordinates";
import { PhaseSpace2D } from "./PhaseSpace2D";
import { RadarChart } from "./RadarChart";
import { StreamgraphChart } from "./StreamgraphChart";

type Level = "gesamt" | "fenster" | "ereignisse";
type WinMin = 5 | 10 | 15;

const VARIABILITY_DE: Record<VariabilityClass, string> = {
  absent: "fehlend",
  minimal: "minimal",
  moderate: "mäßig",
  marked: "ausgeprägt",
};

const LEVEL_TEXT: Record<Level, string> = {
  gesamt:
    "Kennzahlen der ganzen Aufnahme. Die Variabilität ist die Spannweite der gültigen Werte, nicht die Kurzzeitvariabilität.",
  fenster:
    "Gleitende Fenster, Schritt 2 Minuten. Dezelerationen bleiben an den Median der ganzen Aufnahme gebunden.",
  ereignisse:
    "Eine Minute vor dem Beginn und eine Minute nach dem Ende. Eine Wehe zählt nur ab 8 Einheiten über dem Median und 20 Sekunden.",
};

export function AnalysisStation() {
  const [sessionId, setSessionId] = useState(sessions[0].id);
  const session = sessions.find((item) => item.id === sessionId) ?? sessions[0];
  const analysis = useMemo(() => analyzeSession(session), [session]);
  const [level, setLevel] = useState<Level>("gesamt");
  const [winMin, setWinMin] = useState<WinMin>(10);
  const ranges = useMemo(
    () => rollingRanges(session.durationSec, winMin * 60, 120),
    [session.durationSec, winMin],
  );
  const [winIndex, setWinIndex] = useState(0);
  const [eventIndex, setEventIndex] = useState(0);
  const [cursor, setCursor] = useState<number | null>(null);

  useEffect(() => {
    setWinIndex(initialWindow(ranges));
    setCursor(null);
  }, [session.id, winMin, ranges]);

  useEffect(() => {
    setEventIndex(deepestIndex(analysis.decels));
  }, [session.id, analysis.decels]);

  const windowRange = ranges[Math.min(winIndex, Math.max(0, ranges.length - 1))] ?? { t0: 0, t1: session.durationSec };
  const windowScope = useMemo(
    () => summarizeScope(session, windowRange.t0, windowRange.t1, analysis.globalBaseline, analysis.decels, analysis.tocoEvents),
    [session, windowRange.t0, windowRange.t1, analysis],
  );

  const selectedDecel = analysis.decels[eventIndex] ?? null;
  const eventRange = selectedDecel
    ? eventWindow(selectedDecel.t0, selectedDecel.t1, session.durationSec, 60, 60)
    : null;
  const eventScope = useMemo(() => {
    if (!eventRange || !selectedDecel) return null;
    return summarizeScope(session, eventRange.t0, eventRange.t1, analysis.globalBaseline, [selectedDecel], analysis.tocoEvents);
  }, [session, eventRange, selectedDecel, analysis]);

  const scope = level === "fenster" ? windowScope : level === "ereignisse" && eventScope ? eventScope : analysis.global;
  const highlight = level === "fenster" ? windowRange : level === "ereignisse" ? eventRange : null;
  const phaseRange = highlight ?? { t0: 0, t1: session.durationSec };
  const phase = useMemo(
    () => phaseSlice(session.fhr, session.tocoPct, session.dt, phaseRange),
    [session, phaseRange.t0, phaseRange.t1],
  );

  return (
    <div className="min-h-dvh bg-bg text-paper">
      <header className="sticky top-0 z-20 border-b border-line bg-bg/95 backdrop-blur-sm">
        <div className="flex flex-wrap items-end justify-between gap-3 px-4 py-3">
          <div>
            <p className="font-mono text-xs tracking-widest text-copper">CTG ANALYSE</p>
            <h1 className="text-xl font-medium leading-tight">{session.title}</h1>
            <p className="text-sm text-paper/70">
              {session.device} · {session.paper} · Ablesung, kein Befund
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 text-sm">
              <span className="text-paper/70">Aufnahme</span>
              <select
                className="rounded-card border border-line bg-raised px-3 py-2 text-paper"
                value={session.id}
                onChange={(event) => setSessionId(event.target.value)}
              >
                {sessions.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title}
                  </option>
                ))}
              </select>
            </label>
            <Link to="/" className="rounded-card border border-line px-3 py-2 text-sm text-paper/80">
              Streifen
            </Link>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1 border-t border-line px-3 py-2" role="tablist" aria-label="Analyseebene">
          <LevelTab active={level === "gesamt"} onClick={() => setLevel("gesamt")}>
            Gesamt
          </LevelTab>
          <LevelTab active={level === "fenster"} onClick={() => setLevel("fenster")}>
            Fenster
          </LevelTab>
          <LevelTab active={level === "ereignisse"} onClick={() => setLevel("ereignisse")}>
            Ereignisse
          </LevelTab>
          <span className="mx-1 hidden h-6 w-px bg-line sm:block" />
          <Link to="/" className="rounded-card px-3 py-2 text-sm text-paper/70">
            Streifen
          </Link>
          <span className="ml-auto font-mono text-xs text-paper/50">
            {cursor == null ? "" : clockOf(session.startedAt, cursor)}
          </span>
        </div>
      </header>

      <p className="px-4 pt-4 text-sm text-paper/70">{LEVEL_TEXT[level]}</p>
      <MetricStrip items={metricItems(scope)} />

      {level === "fenster" ? (
        <div className="flex flex-wrap items-center gap-3 px-4 pb-2">
          <div className="flex gap-1" role="group" aria-label="Fensterlänge">
            {([5, 10, 15] as const).map((minutes) => (
              <button
                key={minutes}
                type="button"
                onClick={() => setWinMin(minutes)}
                className={
                  winMin === minutes
                    ? "rounded-card border border-copper px-3 py-1.5 text-sm text-copper"
                    : "rounded-card border border-line px-3 py-1.5 text-sm text-paper/60"
                }
              >
                {minutes} min
              </button>
            ))}
          </div>
          <label className="flex min-w-48 flex-1 items-center gap-3 text-sm text-paper/70">
            <span className="shrink-0 font-mono text-xs">
              {clockOf(session.startedAt, windowRange.t0).slice(0, 5)}–{clockOf(session.startedAt, windowRange.t1).slice(0, 5)}
            </span>
            <input
              type="range"
              min={0}
              max={Math.max(0, ranges.length - 1)}
              value={Math.min(winIndex, Math.max(0, ranges.length - 1))}
              onChange={(event) => setWinIndex(Number(event.target.value))}
              aria-label="Fenster"
              className="w-full accent-copper"
            />
          </label>
        </div>
      ) : null}

      {level === "ereignisse" ? (
        <div className="px-4 pb-2">
          <label className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-paper/70">Ereignis</span>
            <select
              className="max-w-full rounded-card border border-line bg-raised px-3 py-2 text-paper"
              value={String(eventIndex)}
              onChange={(event) => setEventIndex(Number(event.target.value))}
              disabled={!analysis.decels.length}
            >
              {analysis.decels.map((event, index) => (
                <option key={`${event.t0}-${index}`} value={index}>
                  {clockOf(session.startedAt, event.troughT).slice(0, 5)} · Tiefe {formatDe(event.depth, 0)} bpm ·{" "}
                  {formatDe(event.durationSec, 0)} s
                </option>
              ))}
            </select>
            <span className="text-xs text-paper/50">
              {analysis.decels.length} Dezelerationen, {analysis.tocoEvents.length} Wehen
            </span>
          </label>
        </div>
      ) : null}

      <ChartFrame
        chart="horizon"
        title="Horizont"
        hint="FHF minus Basis. Grüne Bänder liegen darüber, kupferfarbene darunter, je 15 bpm."
      >
        <HorizonChart
          fhr={session.fhr}
          dt={session.dt}
          durationSec={session.durationSec}
          baseline={analysis.globalBaseline}
          startedAt={session.startedAt}
          highlight={highlight}
          cursorSec={cursor}
          onCursor={setCursor}
        />
      </ChartFrame>

      <ChartFrame chart="barcode" title="Strichcode" hint="Dezelerationen, Lücken ab 5 Sekunden und Kineto-Marken.">
        <BarcodePlot
          durationSec={session.durationSec}
          startedAt={session.startedAt}
          decels={analysis.decels}
          gaps={analysis.gaps}
          events={session.events}
          cursorSec={cursor}
          activeT={level === "ereignisse" ? (selectedDecel?.troughT ?? null) : null}
        />
      </ChartFrame>

      {level !== "ereignisse" ? (
        <div className="grid gap-0 lg:grid-cols-2">
          <ChartFrame chart="radar" title="Radar" hint="Sechs Achsen von 0 bis 100 für den gewählten Ausschnitt.">
            <RadarChart scores={scope.radar} />
          </ChartFrame>
          <ChartFrame chart="phasespace" title="Phasenraum" hint="Aufeinanderfolgende Werte oder Toco gegen FHF. Die Diagonale ist der unveränderte Schlag.">
            <PhaseSpace2D
              fhr={phase.fhr}
              toco={phase.toco}
              dt={session.dt}
              originSec={phase.origin}
              startedAt={session.startedAt}
            />
          </ChartFrame>
        </div>
      ) : (
        <>
          <ChartFrame
            chart="parallel"
            title="Parallelkoordinaten"
            hint="Jede Linie ist eine Dezeleration. Lag ist die Zeit des Tals minus den nächsten Toco-Gipfel."
          >
            <ParallelCoordinates events={analysis.decels} selected={eventIndex} onSelect={setEventIndex} />
          </ChartFrame>
          <ChartFrame chart="phasespace" title="Phasenraum um das Ereignis">
            <PhaseSpace2D
              fhr={phase.fhr}
              toco={phase.toco}
              dt={session.dt}
              originSec={phase.origin}
              startedAt={session.startedAt}
            />
          </ChartFrame>
        </>
      )}

      {level === "gesamt" ? (
        <>
          <ChartFrame chart="stream" title="Zonen" hint="Anteil je 5-Minuten-Block. Die Fläche ist auf die Blockmitte zentriert.">
            <StreamgraphChart fhr={session.fhr} dt={session.dt} durationSec={session.durationSec} startedAt={session.startedAt} />
          </ChartFrame>
          <ChartFrame
            chart="heatmap"
            title="Kreuzkorrelation Zeit × Lag"
            hint="10-Minuten-Fenster, Schritt 2 Minuten. Positiv grün, negativ kupfer. Der helle Rahmen ist der stärkste nicht-positive Lag."
          >
            <CrossCorrHeatmap surface={analysis.surface} startedAt={session.startedAt} />
          </ChartFrame>
        </>
      ) : null}

      {level === "fenster" ? (
        <ChartFrame
          chart="heatmap"
          title="Kreuzkorrelation Zeit × Lag"
          hint="Fest auf 10 Minuten gerechnet, unabhängig von der gewählten Fensterlänge. Die kupferne Zeile ist das aktuelle Fenster, wenn es auf dieses Raster fällt."
        >
          <CrossCorrHeatmap
            surface={analysis.surface}
            startedAt={session.startedAt}
            activeT0={winMin === 10 ? windowRange.t0 : null}
          />
        </ChartFrame>
      ) : null}

      <footer className="border-t border-line px-4 py-4 text-xs leading-relaxed text-paper/55">{SKIPPED_METHODS_NOTE}</footer>
    </div>
  );
}

function metricItems(scope: ScopeSummary): MetricItem[] {
  const klass = scope.variability.klass;
  const amplitude = scope.variability.amplitude;
  return [
    { label: "Basis", value: formatDe(scope.variability.baseline, 0), unit: "bpm" },
    { label: "IQR", value: formatDe(scope.fhr.iqr, 1), unit: "bpm" },
    {
      label: "Spannweite",
      value: klass ? `${VARIABILITY_DE[klass]} · ${formatDe(amplitude, 0)}` : "–",
      unit: klass ? "bpm" : undefined,
    },
    { label: "Schiefe", value: formatDe(scope.fhr.skewness, 2) },
    { label: "Exzess", value: formatDe(scope.fhr.excessKurtosis, 2) },
    { label: "Entropie", value: formatDe(scope.entropy, 2), unit: "bit" },
    { label: "Gzip", value: formatDe(scope.gzipRatio, 2) },
    { label: "|Steigung|", value: formatDe(scope.meanAbsSlope, 2), unit: "bpm/s" },
    { label: "Dezelerationen", value: String(scope.decelerations.length) },
    { label: "AUC", value: formatDe(scope.totalAuc, 0), unit: "bpm·s" },
    { label: "Lag", value: formatSigned(scope.cross.bestLag, 0), unit: "s", accent: true },
    { label: "r FHF–Toco", value: formatDe(scope.cross.bestR, 2) },
    { label: "r FHF–Puls", value: formatDe(scope.fhrMhrR, 2) },
    { label: "Puls", value: formatDe(scope.mhr.median, 0), unit: "bpm" },
    { label: "Toco", value: formatDe(scope.toco.median, 1) },
    {
      label: "Lücken",
      value: formatDe(scope.fhr.nullFraction == null ? null : scope.fhr.nullFraction * 100, 1),
      unit: "%",
    },
    { label: "Periodik", value: scope.periodic ? "auffällig" : "nein" },
  ];
}

function initialWindow(ranges: TimeRange[]): number {
  const index = ranges.findIndex((range) => range.t0 >= 300);
  return index >= 0 ? index : 0;
}

function deepestIndex(events: Deceleration[]): number {
  let best = 0;
  for (let i = 1; i < events.length; i++) {
    if (events[i].depth > events[best].depth) best = i;
  }
  return best;
}

function phaseSlice(
  fhr: readonly (number | null)[],
  toco: readonly (number | null)[],
  dt: number,
  range: TimeRange,
): { fhr: (number | null)[]; toco: (number | null)[]; origin: number } {
  const heart = sliceRange(fhr, dt, range.t0, range.t1);
  const contraction = sliceRange(toco, dt, range.t0, range.t1);
  return { fhr: heart.values, toco: contraction.values, origin: heart.i0 * dt };
}

function LevelTab({ active, onClick, children }: { active: boolean; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={active ? "rounded-card bg-paper px-3 py-2 text-sm text-ink" : "rounded-card px-3 py-2 text-sm text-paper/70"}
    >
      {children}
    </button>
  );
}
