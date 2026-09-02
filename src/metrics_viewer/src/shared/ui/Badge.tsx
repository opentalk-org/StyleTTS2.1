import type { HTMLAttributes, ReactNode } from "react";

import type { RunStatus } from "@/shared/types";
import { cn } from "./cn";

export type BadgeTone = "neutral" | "accent" | "positive" | "negative" | "notice";

const TONES: Record<BadgeTone, string> = {
  neutral: "bg-surface text-fg-secondary border-line",
  accent: "bg-accent-surface text-accent-bright border-accent-border",
  positive: "bg-positive-surface text-positive border-positive-border",
  negative: "bg-negative-surface text-negative border-negative-border",
  notice: "bg-notice-surface text-notice border-notice-border",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;

  icon?: ReactNode;
}

export function Badge({ tone = "neutral", icon, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex w-max items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap",
        TONES[tone],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </span>
  );
}


const STATUS_TONES: Record<RunStatus, BadgeTone> = {
  succeeded: "positive",
  running: "accent",
  failed: "negative",
  awaiting: "notice",
  cancelled: "neutral",
};

const STATUS_MARKS: Record<RunStatus, string> = {
  succeeded: "✓",
  running: "▸",
  failed: "✕",
  awaiting: "•",
  cancelled: "—",
};

export function StatusBadge({ status, className }: { status: RunStatus; className?: string }) {
  return (
    <Badge
      tone={STATUS_TONES[status] ?? "neutral"}
      className={cn("px-1.5 font-mono text-[10px] tracking-tight uppercase", className)}
    >
      <span aria-hidden>{STATUS_MARKS[status] ?? "•"}</span>
      {status}
    </Badge>
  );
}


export function CountPill({ children }: { children: ReactNode }) {
  return (
    <span className="min-w-6 rounded-full bg-surface px-1.5 py-0.5 text-center font-mono text-[10px] tabular-nums text-fg-secondary">
      {children}
    </span>
  );
}
