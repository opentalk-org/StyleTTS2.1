import { Icon } from "../../icons";

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  const clamp = (n: number) => {
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    return n;
  };
  return (
    <div className="flex h-10 items-center overflow-hidden rounded-md border-2 border-transparent bg-panel-2 focus-within:border-blue-500">
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(clamp(parseFloat(e.target.value)))}
        className="h-full w-0 flex-1 border-0 bg-transparent px-3 text-[13.5px] font-semibold text-txt tabular-nums outline-none"
      />
      <div className="flex flex-col border-l border-line">
        <button
          onClick={() => onChange(clamp((value || 0) + step))}
          className="flex h-5 w-[30px] items-center justify-center border-0 border-b border-line bg-panel text-txt-dim cursor-pointer"
        >
          <span className="flex rotate-180">
            <Icon name="chevron-down" size={12} strokeWidth={2.4} />
          </span>
        </button>
        <button
          onClick={() => onChange(clamp((value || 0) - step))}
          className="flex h-5 w-[30px] items-center justify-center border-0 bg-panel text-txt-dim cursor-pointer"
        >
          <Icon name="chevron-down" size={12} strokeWidth={2.4} />
        </button>
      </div>
    </div>
  );
}
