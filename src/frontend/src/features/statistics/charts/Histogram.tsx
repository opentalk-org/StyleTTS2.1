import { BAR_CLASS, type Tone } from "../logic";

/**
 * Vertical bar histogram over pre-binned values. The x-axis shows a row of
 * evenly-spaced tick labels (derived from the real bin edges) plus the unit,
 * so the distribution can be read against an actual scale.
 */
export function Histogram({
  bins,
  ticks,
  tone = "blue",
}: {
  bins: number[];
  ticks: string[];
  tone?: Tone;
}) {
  const max = Math.max(...bins, 0.001);
  return (
    <div>
      <div className="relative flex h-[104px] items-end gap-[2px]">
        {bins.map((v, i) => (
          <div
            key={i}
            className={`flex-1 rounded-t-[2px] opacity-[0.88] ${BAR_CLASS[tone]}`}
            style={{ height: `${Math.max(2, (v / max) * 100)}%` }}
          />
        ))}
      </div>
      <div className="mt-[7px] flex justify-between border-t border-line pt-[6px]">
        {ticks.map((t, i) => (
          <span key={i} className="text-[10px] tabular-nums text-txt-mute">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
