import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";

export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("min-w-0 overflow-hidden rounded-md border border-line bg-surface", className)}
      {...rest}
    >
      {children}
    </div>
  );
}

export function CardHeader({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex h-card-head flex-none items-center justify-between gap-2 border-b border-line px-3",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

/** Secondary 12px label: column headers, section eyebrows, captions. */
export function Caption({ className, children, ...rest }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span className={cn("text-xs font-medium text-fg-muted", className)} {...rest}>
      {children}
    </span>
  );
}

/** Numeric value in the mono face with tabular figures. */
export function Numeric({ className, children, ...rest }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span className={cn("font-mono text-xs tabular-nums", className)} {...rest}>
      {children}
    </span>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-4.5 min-w-4.5 items-center justify-center rounded-sm border border-strong bg-inset px-1 font-mono text-[11px] leading-none text-fg-secondary">
      {children}
    </kbd>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <span aria-hidden className={cn("block animate-pulse rounded-sm bg-hover", className)} />;
}

/** Thin indeterminate progress line for "fetching" states at the top of a pane. */
export function ProgressLine({ active }: { active: boolean }) {
  return (
    <span aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-0.5 overflow-hidden">
      {active ? <span className="progress-slide block h-full w-1/3 bg-accent" /> : null}
    </span>
  );
}
