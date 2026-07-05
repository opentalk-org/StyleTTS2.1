/** Range slider with a tabular value readout. */
export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  format = (v) => String(v),
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (value: number) => string;
}) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 cursor-pointer accent-blue-500"
      />
      <span className="min-w-10 text-right text-[12.5px] font-bold text-blue-600 tabular-nums">
        {format(value)}
      </span>
    </div>
  );
}
