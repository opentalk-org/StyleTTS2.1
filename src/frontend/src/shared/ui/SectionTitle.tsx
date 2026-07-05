import type { ReactNode } from "react";

import { cn } from "./cn";

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
