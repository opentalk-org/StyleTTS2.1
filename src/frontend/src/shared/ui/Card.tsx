import type { ReactNode } from "react";

import { cn } from "./cn";

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
