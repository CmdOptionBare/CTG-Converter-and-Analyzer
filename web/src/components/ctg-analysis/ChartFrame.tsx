import type { ReactNode } from "react";

export function ChartFrame({
  title,
  hint,
  chart,
  children,
}: {
  title: string;
  hint?: string;
  chart?: string;
  children: ReactNode;
}) {
  return (
    <section className="border-t border-line px-4 py-4" data-chart={chart}>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 className="text-sm font-medium">{title}</h2>
        {hint ? <p className="max-w-2xl text-xs leading-relaxed text-paper/50">{hint}</p> : null}
      </div>
      {children}
    </section>
  );
}

export function HoverTip({ x, y, text }: { x: number; y: number; text: string }) {
  return (
    <div
      className="pointer-events-none absolute z-10 whitespace-pre rounded-card border border-line bg-raised px-2 py-1 font-mono text-[11px] leading-snug text-paper"
      style={{ left: Math.max(8, x), top: Math.max(8, y), transform: "translate(12px, -110%)" }}
    >
      {text}
    </div>
  );
}
