import type { ReactNode } from "react";

import { cn } from "./cn";

export interface SegmentedOption<T extends string | number> {
  value: T;
  label: ReactNode;
  title?: string;
  /** Greys out this choice alone; the rest of the control stays usable. */
  disabled?: boolean;
}

export interface SegmentedControlProps<T extends string | number> {
  label: string;
  options: SegmentedOption<T>[];
  value: T;
  onValue: (value: T) => void;
  leading?: ReactNode;
  fill?: boolean;
  disabled?: boolean;
  className?: string;
}

/**
 * Neutral selection style on purpose: the accent is reserved for row selection and
 * primary actions so a segmented control never competes with them.
 */
export function SegmentedControl<T extends string | number>({
  label,
  options,
  value,
  onValue,
  leading,
  fill = false,
  disabled = false,
  className,
}: SegmentedControlProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn(
        "flex h-control items-center gap-0.5 rounded-md border border-strong bg-inset p-0.5",
        leading === undefined ? "" : "pl-2",
        fill ? "w-full" : "",
        disabled ? "opacity-50" : "",
        className,
      )}
    >
      {leading === undefined ? null : <span className="mr-1 text-fg-muted">{leading}</span>}
      {options.map((option) => {
        const selected = option.value === value;
        const off = disabled || option.disabled === true;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            title={option.title}
            disabled={off}
            onClick={() => onValue(option.value)}
            className={cn(
              "flex h-full min-w-6 items-center justify-center gap-1 rounded-sm px-2 text-xs font-medium",
              "transition-[background-color,color] duration-100 ease-out",
              fill ? "min-w-0 flex-1" : "",
              selected ? "bg-raised text-fg shadow-[inset_0_0_0_1px_var(--color-strong)]" : "text-fg-muted hover:text-fg",
              option.disabled === true && !disabled ? "opacity-45" : "",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
