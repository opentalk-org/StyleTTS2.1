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
        "flex flex-col items-center justify-center gap-1.5 px-6 text-center",
        compact ? "py-10" : "min-h-72 flex-1",
        className,
      )}
    >
      <span className="mb-1 text-fg-muted [&>svg]:size-5">{icon}</span>
      <h3 className="text-[13px] font-medium text-fg">{title}</h3>
      {description === undefined ? null : (
        <p className="max-w-md text-xs leading-relaxed text-fg-muted">{description}</p>
      )}
      {children === undefined ? null : <div className="mt-2 flex items-center gap-2">{children}</div>}
    </div>
  );
}
