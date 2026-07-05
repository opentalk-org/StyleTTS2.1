import { BAR_CLASS, type RankItem, type Tone } from "../logic";

/**
 * Ranked rows: index, monospaced token chip, proportional bar, and count.
 * Used for the top / bottom trigram lists.
 */
export function RankList({ items, tone = "blue" }: { items: RankItem[]; tone?: Tone }) {
  const max = Math.max(...items.map((it) => it.value), 0.001);
  return (
    <div className="flex flex-col gap-[6px]">
      {items.map((it, i) => (
        <div key={i} className="grid grid-cols-[18px_70px_1fr_52px] items-center gap-[9px]">
          <span className="text-[10.5px] font-bold tabular-nums text-txt-mute">{i + 1}</span>
          <span className="whitespace-pre rounded-[4px] bg-panel-2 px-[6px] py-[2px] text-center font-mono text-[12px] text-txt">
            {it.label}
          </span>
          <div className="h-[8px] overflow-hidden rounded-[4px] bg-panel-2">
            <div
              className={`h-full rounded-[4px] ${BAR_CLASS[tone]}`}
              style={{ width: `${(it.value / max) * 100}%` }}
            />
          </div>
          <span className="text-right text-[11px] font-semibold tabular-nums text-txt-dim">
            {it.value.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
}
