import type { ReactNode } from "react";

import { cn } from "./cn";

export interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  description?: ReactNode;

  compact?: boolean;
  children?: ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, compact = false, children, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-line px-6 text-center",
        compact ? "h-auto min-h-40 py-8" : "h-full min-h-80",
        className,
      )}
    >
      <span className="mb-1 text-fg-muted [&>svg]:size-6">{icon}</span>
      <h3 className="m-0 text-sm font-medium text-fg-secondary">{title}</h3>
      {description === undefined ? null : (
        <p className="m-0 max-w-md text-xs leading-relaxed text-fg-muted">{description}</p>
      )}
      {children}
    </div>
  );
}
