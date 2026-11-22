export function clockOf(startedAt: string, tSec: number): string {
  const match = /^(\d{2}):(\d{2}):(\d{2})/.exec(startedAt.slice(11, 19));
  const h = match ? Number(match[1]) : 0;
  const m = match ? Number(match[2]) : 0;
  const s = match ? Number(match[3]) : 0;
  const total = h * 3600 + m * 60 + s + Math.max(0, Math.round(tSec));
  const hh = Math.floor(total / 3600) % 24;
  const mm = Math.floor(total / 60) % 60;
  const ss = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hh)}:${pad(mm)}:${pad(ss)}`;
}

export function formatDe(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return new Intl.NumberFormat("de-DE", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatSigned(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "–";
  const text = formatDe(Math.abs(value), digits);
  if (value > 0) return `+${text}`;
  if (value < 0) return `−${text}`;
  return text;
}
