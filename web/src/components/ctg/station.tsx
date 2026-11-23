import { useEffect, useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import { Download, Minus, Plus } from "lucide-react";
import { sessions } from "@/data/sessions";
import { clockOf, formatSample, tocoKpa, toCsv, windowStats } from "@/lib/ctg/analyze";
import { CtgPaper } from "@/components/ctg/paper";

const ZOOMS = [18, 26, 36];

export function Station() {
  const [sessionId, setSessionId] = useState(sessions[0].id);
  const session = sessions.find((s) => s.id === sessionId) ?? sessions[0];
  const [zoom, setZoom] = useState(1);
  const [showPhoto, setShowPhoto] = useState(false);
  const [opacity, setOpacity] = useState(0.55);
  const [showFhr, setShowFhr] = useState(true);
  const [showMhr, setShowMhr] = useState(true);
  const [showToco, setShowToco] = useState(true);
  const [tab, setTab] = useState<"streifen" | "werte" | "pipeline">("streifen");
  const [cursor, setCursor] = useState<number | null>(null);
  const [jump, setJump] = useState<{ t: number; n: number } | undefined>(undefined);
  const [scanFile, setScanFile] = useState(session.scans[0]?.file ?? "");

  const firstScan = session.scans[0]?.file ?? "";
  useEffect(() => {
    setScanFile(firstScan);
    setCursor(null);
    setJump(undefined);
  }, [session.id, firstScan]);

  const px = ZOOMS[zoom];
  const t = cursor ?? 0;
  const i = session.fhr.length
    ? Math.min(session.fhr.length - 1, Math.max(0, Math.round(t / session.dt)))
    : -1;
  const stats = useMemo(() => windowStats(session), [session]);
  const factor = session.tocoKpaPerPct;
  const fhrNow = i >= 0 ? session.fhr[i] : null;
  const mhrNow = i >= 0 ? session.mhr[i] : null;
  const tocoNow = i >= 0 ? session.tocoPct[i] : null;
  const tocoKpaNow = tocoKpa(tocoNow, factor);

  function download() {
    const blob = new Blob([toCsv(session)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ctg-${session.id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function go(sec: number) {
    setJump({ t: sec, n: Date.now() });
  }

  return (
    <div className="flex h-dvh flex-col bg-bg text-paper">
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-line px-4 py-3">
        <div>
          <p className="font-mono text-xs tracking-widest text-copper">CTG BLATT</p>
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
              onChange={(e) => setSessionId(e.target.value)}
            >
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.title}
                </option>
              ))}
            </select>
          </label>
          <Link to="/analyse" className="rounded-card border border-copper px-3 py-2 text-sm text-copper">
            Analyse
          </Link>
        </div>
      </header>

      <div className="flex gap-2 overflow-x-auto border-b border-line px-3 py-2">
        <Readout label="Zeit" value={clockOf(session.startedAt, cursor ?? 0)} />
        <Readout label="FHF" value={showFhr ? formatSample(fhrNow, 0) : "–"} unit="bpm" />
        <Readout label="Puls" value={showMhr ? formatSample(mhrNow, 0) : "–"} unit="bpm" accent />
        <Readout
          label="Toco"
          value={showToco ? formatSample(tocoNow, 1) : "–"}
          hint={showToco && tocoKpaNow != null ? `${tocoKpaNow.toFixed(1)} kPa` : undefined}
        />
      </div>

      <div className={tab === "streifen" ? "flex min-h-0 flex-1 flex-col" : "hidden"}>
        <div className="min-h-0 flex-1 overflow-y-auto py-2">
          <CtgPaper
            session={session}
            pxPerCm={px}
            showPhoto={showPhoto}
            photoOpacity={opacity}
            showFhr={showFhr}
            showMhr={showMhr}
            showToco={showToco}
            onCursor={setCursor}
            scrollToSec={jump?.t}
            scrollToken={jump?.n}
          />
        </div>
        <div className="flex gap-1 overflow-x-auto border-t border-line px-3 py-2">
          {Array.from({ length: Math.ceil(session.durationSec / 60) }, (_, m) => (
            <button
              key={m}
              type="button"
              onClick={() => go(m * 60)}
              className="shrink-0 rounded-card border border-line px-2 py-1 font-mono text-xs text-paper/80"
            >
              {clockOf(session.startedAt, m * 60).slice(0, 5)}
            </button>
          ))}
        </div>
      </div>

      {tab === "werte" && (
        <div className="min-h-0 flex-1 overflow-auto px-4 py-4">
          <p className="mb-3 max-w-2xl text-sm text-paper/75">{session.provenance.note}</p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] text-left text-sm">
              <thead className="text-xs text-paper/60">
                <tr>
                  <th className="py-2 pr-3 font-medium">Zeit</th>
                  <th className="py-2 pr-3 font-medium">Basis</th>
                  <th className="py-2 pr-3 font-medium">Band p10–p90</th>
                  <th className="py-2 pr-3 font-medium">Anstiege</th>
                  <th className="py-2 pr-3 font-medium">Abfälle</th>
                  <th className="py-2 pr-3 font-medium">Puls</th>
                  <th className="py-2 font-medium">Toco 0–100</th>
                </tr>
              </thead>
              <tbody>
                {stats.map((w) => (
                  <tr key={w.t0} className="border-t border-line font-mono">
                    <td className="py-2 pr-3">
                      {clockOf(session.startedAt, w.t0).slice(0, 5)}–{clockOf(session.startedAt, w.t1).slice(0, 5)}
                    </td>
                    <td className="py-2 pr-3">{w.baseline == null ? "–" : w.baseline}</td>
                    <td className="py-2 pr-3">
                      {w.low == null || w.high == null ? "–" : `${w.low}–${w.high} (${w.band})`}
                    </td>
                    <td className="py-2 pr-3">{w.rises}</td>
                    <td className="py-2 pr-3">{w.falls}</td>
                    <td className="py-2 pr-3">{w.mhr == null ? "–" : w.mhr}</td>
                    <td className="py-2">{tocoCell(w.toco, factor)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 max-w-2xl text-xs text-paper/60">
            Anstieg: mindestens 15 Schläge über der Basis, mindestens 15 Sekunden. Abfall entsprechend darunter.
            Fehlende Stellen zählen nicht mit. Gerechnet nur aus der Serie, nicht aus dem Foto. Toco ist die
            Skala 0 bis 100. Ein kPa-Wert steht nur daneben, wenn die Aufnahme einen Umrechnungsfaktor enthält.
          </p>
          <h2 className="mb-2 mt-6 text-sm font-medium">Aufdruck auf dem Papier</h2>
          <ul className="space-y-1 font-mono text-sm">
            {session.events.map((e, n) => (
              <li key={`${e.t}-${n}`}>
                <span className="text-paper/60">{clockOf(session.startedAt, e.t).slice(0, 5)}</span> {e.text}
              </li>
            ))}
          </ul>
        </div>
      )}

      {tab === "pipeline" && <Pipeline />}

      <footer className="flex flex-wrap items-center gap-2 border-t border-line px-3 py-2">
        <TabButton active={tab === "streifen"} onClick={() => setTab("streifen")}>
          Streifen
        </TabButton>
        <TabButton active={tab === "werte"} onClick={() => setTab("werte")}>
          Werte
        </TabButton>
        <TabButton active={tab === "pipeline"} onClick={() => setTab("pipeline")}>
          Pipeline
        </TabButton>
        <Link to="/analyse" className="rounded-card px-3 py-2 text-sm text-paper/70">
          Analyse
        </Link>
        <span className="mx-1 hidden h-6 w-px bg-line sm:block" />
        <Toggle on={showFhr} onClick={() => setShowFhr((v) => !v)}>
          FHF
        </Toggle>
        <Toggle on={showMhr} onClick={() => setShowMhr((v) => !v)}>
          Puls
        </Toggle>
        <Toggle on={showToco} onClick={() => setShowToco((v) => !v)}>
          Toco
        </Toggle>
        <Toggle on={showPhoto} onClick={() => setShowPhoto((v) => !v)}>
          Original
        </Toggle>
        {session.scans.length > 0 && (
          <select
            className="max-w-[16rem] rounded-card border border-line bg-raised px-2 py-2 text-xs text-paper"
            value={scanFile}
            onChange={(e) => {
              const file = e.target.value;
              setScanFile(file);
              const scan = session.scans.find((s) => s.file === file);
              if (scan) go(scan.t0);
            }}
            aria-label="Bild"
          >
            {session.scans.map((s, n) => (
              <option key={`${s.file}-${n}`} value={s.file}>
                {String(n + 1).padStart(2, "0")} {clockOf(session.startedAt, s.t0).slice(0, 5)}–
                {clockOf(session.startedAt, s.t1).slice(0, 5)} · {s.trust}
              </option>
            ))}
          </select>
        )}
        {showPhoto && (
          <label className="flex items-center gap-2 text-xs text-paper/70">
            Deckkraft
            <input
              type="range"
              min={0.15}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              aria-label="Deckkraft"
            />
          </label>
        )}
        <span className="flex-1" />
        <button
          type="button"
          className="rounded-card border border-line p-2"
          onClick={() => setZoom((z) => Math.max(0, z - 1))}
          aria-label="Kleiner"
        >
          <Minus size={16} />
        </button>
        <button
          type="button"
          className="rounded-card border border-line p-2"
          onClick={() => setZoom((z) => Math.min(ZOOMS.length - 1, z + 1))}
          aria-label="Größer"
        >
          <Plus size={16} />
        </button>
        <button
          type="button"
          className="flex items-center gap-2 rounded-card bg-copper px-3 py-2 text-sm text-ink"
          onClick={download}
        >
          <Download size={16} />
          CSV
        </button>
      </footer>
    </div>
  );
}

function tocoCell(pct: number | null, factor: number | undefined): string {
  if (pct == null) return "–";
  const kpa = tocoKpa(pct, factor);
  const text = pct.toFixed(1);
  return kpa == null ? text : `${text} (${kpa.toFixed(1)} kPa)`;
}

function Readout({
  label,
  value,
  unit,
  hint,
  accent,
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <div className="min-w-24 rounded-card bg-raised px-3 py-2">
      <div className="text-xs text-paper/55">{label}</div>
      <div className={accent ? "font-mono text-lg text-copper" : "font-mono text-lg"}>
        {value}
        {unit && value !== "–" && <span className="ml-1 text-xs text-paper/50">{unit}</span>}
        {hint && value !== "–" && <span className="ml-1 text-xs text-paper/50">{hint}</span>}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        active ? "rounded-card bg-paper px-3 py-2 text-sm text-ink" : "rounded-card px-3 py-2 text-sm text-paper/70"
      }
    >
      {children}
    </button>
  );
}

function Toggle({ on, onClick, children }: { on: boolean; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={
        on
          ? "rounded-card border border-copper px-3 py-2 text-sm text-copper"
          : "rounded-card border border-line px-3 py-2 text-sm text-paper/50"
      }
    >
      {children}
    </button>
  );
}

const STEPS = [
  ["Fotos ordnen", "Mensch", "Die Aufnahmen einer Rolle in zeitlicher Reihenfolge. Daraus wird ein Datensatz."],
  ["Bild aufbereiten", "Rechner", "Belichtung glätten, grünes Raster und dunkle Kurve trennen. Kein Modell."],
  [
    "Blatt entzerren",
    "Rechner",
    "Das Raster wird auf 90 Grad und auf 100 Pixel pro Zentimeter gezogen. Philips M1911A, ein Zentimeter je Minute.",
  ],
  ["Panorama fügen", "Rechner", "Überlappung messen und die Bilder zu einem Streifen zusammensetzen."],
  [
    "Kurven folgen",
    "Rechner",
    "FHF, Puls und Toco als Linie auf dem Panorama, eine Zahl pro Sekunde. Keine KI, keine abgelesenen Stützstellen.",
  ],
  ["Serie schreiben", "Rechner", "JSON und CSV. Diese Seite liest nur die Serie und das Panorama."],
  [
    "Blatt zeichnen",
    "Rechner",
    "Dieselben Maße wie das Papier: 8 cm für 50 bis 210 Schläge, 4 cm für die Toco-Skala 0 bis 100, ein Zentimeter je Minute.",
  ],
  [
    "Gegencheck",
    "Mensch",
    "Original einblenden. Die gezeichnete Linie soll auf der Druckfarbe liegen, das Raster auf dem fotografierten Raster.",
  ],
];

function Pipeline() {
  return (
    <div className="min-h-0 flex-1 overflow-auto px-4 py-4">
      <p className="mb-4 max-w-2xl text-sm text-paper/75">
        Neue Aufnahmen kommen aus der Bildpipeline dazu. Entzerren, Zusammenfügen und Ablesen sind festgelegt; eine
        KI liest keine Stützstellen. Die Seite zeichnet nur, was die Pipeline geschrieben hat.
      </p>
      <ol className="max-w-2xl space-y-3">
        {STEPS.map(([title, who, text], n) => (
          <li key={title} className="flex gap-3">
            <span className="font-mono text-sm text-copper">{String(n + 1).padStart(2, "0")}</span>
            <div>
              <div className="flex flex-wrap items-baseline gap-2">
                <span className="font-medium">{title}</span>
                <span className="font-mono text-xs text-paper/50">{who}</span>
              </div>
              <p className="text-sm text-paper/70">{text}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
