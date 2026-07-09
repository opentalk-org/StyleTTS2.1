import type { ReactNode } from "react";

import { Icon, type IconName } from "../icons";

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: IconName;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-5 py-16 text-center">
      <div className="flex h-13 w-13 items-center justify-center rounded-xl bg-panel-2 p-3.5 text-txt-mute">
        <Icon name={icon} size={24} strokeWidth={1.8} />
      </div>
      <div className="text-sm font-semibold text-txt-dim">{title}</div>
      {description ? <div className="text-[13px] text-txt-mute">{description}</div> : null}
      {action}
    </div>
  );
}
