import type { ReactNode } from "react";

/** Titled card grouping a set of settings rows. */
export function SettingsSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="mb-4 rounded-[10px] border border-line bg-panel px-5 pb-2 pt-1.5">
      <div className="pt-3.5 pb-0.5 text-[11px] font-bold uppercase tracking-wider text-blue-500">
        {title}
      </div>
      {children}
    </div>
  );
}
