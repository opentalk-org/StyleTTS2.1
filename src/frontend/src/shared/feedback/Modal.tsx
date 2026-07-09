import type { ReactNode } from "react";

import { Icon, type IconName } from "../icons";
import { cn } from "../ui/cn";

export function Modal({
  icon,
  title,
  desc,
  danger = false,
  onClose,
  footer,
  children,
  maxWidth = 480,
}: {
  icon: IconName;
  title: string;
  desc?: string;
  danger?: boolean;
  onClose: () => void;
  footer?: ReactNode;
  children?: ReactNode;
  maxWidth?: number;
}) {
  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-[300] flex items-center justify-center bg-gray-900/45 p-6"
      style={{ animation: "fadein 140ms ease" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth }}
        className="flex max-h-[88vh] w-full flex-col overflow-hidden rounded-xl bg-panel"
      >
        <div className="flex items-start gap-3 px-[22px] pt-5 pb-4">
          <div
            className={cn(
              "flex h-[38px] w-[38px] flex-none items-center justify-center rounded-[9px]",
              danger ? "bg-red-50 text-red-500" : "bg-blue-50 text-blue-600",
            )}
          >
            <Icon name={icon} size={19} strokeWidth={2.2} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-[17px] font-bold tracking-tight text-txt">{title}</div>
            {desc ? (
              <div className="mt-1 text-[13px] leading-normal text-txt-dim">{desc}</div>
            ) : null}
          </div>
          <button
            onClick={onClose}
            className="flex h-[30px] w-[30px] flex-none items-center justify-center rounded-md border-0 bg-transparent text-txt-mute cursor-pointer hover:bg-panel-2"
          >
            <Icon name="x" size={17} strokeWidth={2.4} />
          </button>
        </div>
        {children ? <div className="overflow-y-auto px-[22px] pb-5 pt-1">{children}</div> : null}
        {footer ? (
          <div className="flex gap-2.5 border-t border-line bg-panel-2 px-[22px] py-4">
            {footer}
          </div>
        ) : null}
      </div>
    </div>
  );
}
