import type { ReactNode } from "react";

export function SettingsRow({
  title,
  desc,
  children,
}: {
  title: string;
  desc: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-center gap-4 border-b border-line py-4 last:border-b-0">
      <div className="flex-1">
        <div className="text-[13.5px] font-semibold text-txt">{title}</div>
        <div className="mt-0.5 text-xs text-txt-mute">{desc}</div>
      </div>
      {children}
    </div>
  );
}
