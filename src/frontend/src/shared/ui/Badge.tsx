import type { ReactNode } from "react";

import { cn } from "./cn";

export type Tone = "blue" | "emerald" | "amber" | "red" | "gray";

const TONES: Record<Tone, string> = {
  blue: "bg-blue-50 text-blue-700",
  emerald: "bg-emerald-50 text-emerald-700",
  amber: "bg-amber-50 text-amber-700",
  red: "bg-red-50 text-red-600",
  gray: "bg-panel-2 text-txt-mute",
};

export function Badge({
  tone = "gray",
  shape = "chip",
  className,
  children,
}: {
  tone?: Tone;
  shape?: "chip" | "pill";
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-bold whitespace-nowrap",
        TONES[tone],
        shape === "chip"
          ? "h-5 rounded px-1.5 text-[10px] uppercase tracking-wide"
          : "h-[22px] rounded-full px-2.5 text-[11px]",
        className,
      )}
    >
      {children}
    </span>
  );
}
