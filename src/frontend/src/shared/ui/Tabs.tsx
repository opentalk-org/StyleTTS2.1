import type { Option } from "./Select";
import { cn } from "./cn";

export function Tabs({
  value,
  onChange,
  options,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Option[];
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex w-fit gap-1.5 rounded-lg border border-line bg-panel p-[5px]",
        className,
      )}
    >
      {options.map((o) => {
        const on = value === o.value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={cn(
              "h-[34px] rounded-md border-0 px-4 text-[13px] font-semibold cursor-pointer transition-colors",
              on ? "bg-blue-500 text-white" : "bg-transparent text-txt-dim",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
