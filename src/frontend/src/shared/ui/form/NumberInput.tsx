export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  decimals,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  decimals?: number;
}) {
  const clamp = (n: number) => {
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    return n;
  };
  return (
    <div className="flex h-10 items-center overflow-hidden rounded-md border-2 border-line bg-panel focus-within:border-blue-500">
      <input
        type="number"
        value={decimals === undefined ? value : value.toFixed(decimals)}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(clamp(parseFloat(e.target.value)))}
        className="h-full w-full appearance-none border-0 bg-transparent px-3 text-[13.5px] font-semibold text-txt tabular-nums outline-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      />
    </div>
  );
}
