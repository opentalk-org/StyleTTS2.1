import { BAR_CLASS, type HBarItem, type Tone } from "../logic";

/**
 * Horizontal labeled bars: a fixed label column, a proportional bar, and a
 * right-aligned value. Shared by per-speaker duration/length and 1-gram
 * frequency charts.
 */
export function HBars({ items, tone = "emerald" }: { items: HBarItem[]; tone?: Tone }) {
  const max = Math.max(...items.map((it) => it.value), 0.001);
  return (
    <div className="flex flex-col gap-[9px]">
      {items.map((it, i) => (
        <div key={i} className="grid grid-cols-[104px_1fr_64px] items-center gap-[10px]">
          <span className="truncate text-[12px] font-medium text-txt-dim">{it.label}</span>
          <div className="h-[18px] overflow-hidden rounded-[4px] bg-panel-2">
            <div
              className={`h-full rounded-[4px] ${BAR_CLASS[tone]}`}
              style={{ width: `${(it.value / max) * 100}%` }}
            />
          </div>
          <span className="text-right text-[11.5px] font-semibold tabular-nums text-txt">
            {it.display}
          </span>
        </div>
      ))}
    </div>
  );
}
