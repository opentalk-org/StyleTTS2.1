import { BAR_CLASS, type Tone } from "../logic";

/**
 * Vertical bar histogram over pre-binned, [0,1]-normalized values. The x-axis
 * shows only min / mid / max labels plus the unit — enough to read the shape of
 * a distribution without a full scale.
 */
export function Histogram({
  bins,
  xmin,
  xmid,
  xmax,
  tone = "blue",
}: {
  bins: number[];
  xmin: string;
  xmid: string;
  xmax: string;
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
        {[xmin, xmid, xmax].map((t, i) => (
          <span key={i} className="text-[10px] tabular-nums text-txt-mute">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
