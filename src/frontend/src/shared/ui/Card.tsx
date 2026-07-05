import type { ReactNode } from "react";

import { cn } from "./cn";

/** White panel with the standard 1px line border and rounded corners. */
export function Card({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={cn("rounded-[10px] border border-line bg-panel", className)}>
      {children}
    </div>
  );
}

/** Uppercase muted section label. */
export function SectionTitle({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "text-[13px] font-bold uppercase tracking-wider text-txt-mute",
        className,
      )}
    >
      {children}
    </div>
  );
}
