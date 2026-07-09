import { Icon } from "../icons";
import { cn } from "./cn";

export type Option = { value: string; label: string };

export function Select({
  value,
  onChange,
  options,
  variant = "filled",
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Option[];
  variant?: "mini" | "filled";
  className?: string;
}) {
  const mini = variant === "mini";
  return (
    <div className={cn("relative", mini ? "" : "w-full", className)}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={cn(
          "w-full appearance-none rounded-md pr-8 pl-3 font-semibold text-txt outline-none cursor-pointer transition-colors",
          mini
            ? "h-9 bg-panel border border-line text-[12.5px] focus:border-blue-500"
            : "h-10 bg-panel-2 border-2 border-transparent text-[13.5px] focus:border-blue-500",
        )}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <span className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-txt-mute">
        <Icon name="chevron-down" size={13} strokeWidth={2.2} />
      </span>
    </div>
  );
}
