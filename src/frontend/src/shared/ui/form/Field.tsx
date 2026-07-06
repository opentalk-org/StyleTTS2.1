import type { ReactNode } from "react";

/** Labeled form control with an optional hint line. */
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-semibold text-txt">{label}</span>
      {children}
      {hint ? <span className="text-[11px] text-txt-mute">{hint}</span> : null}
    </label>
  );
}
