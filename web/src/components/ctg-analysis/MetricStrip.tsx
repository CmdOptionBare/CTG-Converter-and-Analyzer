export type MetricItem = {
  label: string;
  value: string;
  unit?: string;
  accent?: boolean;
};

export function MetricStrip({ items }: { items: MetricItem[] }) {
  return (
    <div className="flex gap-2 overflow-x-auto px-4 py-3" data-chart="metrics">
      {items.map((item) => (
        <div key={item.label} className="min-w-24 shrink-0 rounded-card bg-raised px-3 py-2">
          <div className="text-xs text-paper/55">{item.label}</div>
          <div className={item.accent ? "font-mono text-lg text-copper" : "font-mono text-lg"}>
            {item.value}
            {item.unit && item.value !== "–" ? <span className="ml-1 text-xs text-paper/50">{item.unit}</span> : null}
          </div>
        </div>
      ))}
    </div>
  );
}
