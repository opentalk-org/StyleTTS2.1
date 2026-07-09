import type { Option } from "../Select";
import { cn } from "../cn";

export function RadioGroup({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Option[];
}) {
  return (
    <div className="flex gap-1 rounded-md bg-panel-2 p-1">
      {options.map((o) => {
        const on = value === o.value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={cn(
              "h-8 flex-1 rounded border-0 text-[12.5px] font-semibold cursor-pointer transition-colors",
              on ? "bg-panel text-txt" : "bg-transparent text-txt-dim",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
