import type { HTMLAttributes } from "react";

import { cn } from "./cn";


export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "min-w-0 overflow-hidden rounded-xl border border-line bg-elevated shadow-card",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}


export function CardHeader({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("flex h-12 items-center justify-between gap-3 border-b border-line px-3", className)}
      {...rest}
    >
      {children}
    </div>
  );
}


export function GroupLabel({ className, children, ...rest }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn("text-[10px] font-medium tracking-widest text-fg-muted uppercase", className)}
      {...rest}
    >
      {children}
    </span>
  );
}
