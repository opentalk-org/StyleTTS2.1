import { Check, Circle, Minus, X } from "lucide-react";
import type { HTMLAttributes, ReactNode } from "react";

import type { RunStatus } from "@/shared/types";
import { cn } from "./cn";

export type BadgeTone = "neutral" | "accent" | "positive" | "negative" | "notice";

const TONES: Record<BadgeTone, string> = {
  neutral: "bg-hover text-fg-secondary",
  accent: "bg-accent-subtle text-accent",
  positive: "bg-succeeded-bg text-succeeded",
  negative: "bg-failed-bg text-failed",
  notice: "bg-queued-bg text-queued",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  icon?: ReactNode;
}

export function Badge({ tone = "neutral", icon, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex h-5 w-max items-center gap-1 rounded-sm px-1.5 text-xs font-medium whitespace-nowrap",
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

const STATUS_CLASSES: Record<RunStatus, string> = {
  running: "bg-running-bg text-running",
  succeeded: "bg-succeeded-bg text-succeeded",
  failed: "bg-failed-bg text-failed",
  queued: "bg-queued-bg text-queued",
  cancelled: "bg-cancelled-bg text-cancelled",
};

const STATUS_LABELS: Record<RunStatus, string> = {
  running: "Running",
  succeeded: "Succeeded",
  failed: "Failed",
  queued: "Queued",
  cancelled: "Cancelled",
};

export function StatusMark({ status, className }: { status: RunStatus; className?: string }) {
  const size = 11;
  if (status === "running") {
    return (
      <Circle
        size={8}
        fill="currentColor"
        strokeWidth={0}
        className={cn("status-pulse shrink-0", className)}
        aria-hidden
      />
    );
  }
  if (status === "succeeded") return <Check size={size} className={cn("shrink-0", className)} aria-hidden />;
  if (status === "failed") return <X size={size} className={cn("shrink-0", className)} aria-hidden />;
  if (status === "queued") return <Circle size={8} className={cn("shrink-0", className)} aria-hidden />;
  return <Minus size={size} className={cn("shrink-0", className)} aria-hidden />;
}

export function StatusBadge({ status, className }: { status: RunStatus; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-5 w-max items-center gap-1.5 rounded-sm px-1.5 text-xs font-medium whitespace-nowrap",
        STATUS_CLASSES[status],
        className,
      )}
    >
      <StatusMark status={status} />
      {STATUS_LABELS[status]}
    </span>
  );
}

export function statusLabel(status: RunStatus): string {
  return STATUS_LABELS[status];
}
