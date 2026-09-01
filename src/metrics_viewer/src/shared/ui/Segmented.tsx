import type { ReactNode } from "react";

import { cn } from "./cn";

export interface SegmentedOption<T extends string | number> {
  value: T;
  label: ReactNode;
  /** Accessible name when `label` is only an icon or a bare digit. */
  title?: string;
}

export interface SegmentedControlProps<T extends string | number> {
  label: string;
  options: SegmentedOption<T>[];
  value: T;
  onValue: (value: T) => void;
  /** Rendered inside the container, before the segments. */
  leading?: ReactNode;
  /** Segments share the full width instead of hugging their labels. */
  fill?: boolean;
  className?: string;
}

/** Inset container, transparent segments, one slightly elevated active segment. */
export function SegmentedControl<T extends string | number>({
  label,
  options,
  value,
  onValue,
  leading,
  fill = false,
  className,
}: SegmentedControlProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn(
        "flex h-8 items-center gap-0.5 rounded-lg border border-line bg-inset p-0.5",
        leading === undefined ? "" : "pl-2",
        fill ? "w-full" : "",
        className,
      )}
    >
      {leading === undefined ? null : <span className="mr-1 text-fg-muted">{leading}</span>}
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            title={option.title}
            onClick={() => onValue(option.value)}
            className={cn(
              "flex h-7 min-w-7 items-center justify-center gap-1.5 rounded-[6px] px-2 text-xs font-medium",
              "transition-[background-color,color,box-shadow] duration-150 ease-out",
              fill ? "min-w-0 flex-1" : "",
              selected
                ? "bg-accent-surface text-accent-bright shadow-[inset_0_0_0_1px_rgb(99_102_241/0.18)]"
                : "text-fg-muted hover:bg-surface hover:text-fg-secondary",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
