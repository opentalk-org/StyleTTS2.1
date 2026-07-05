import type { ButtonHTMLAttributes } from "react";

import { Icon, type IconName } from "../icons";
import { cn } from "./cn";

/** Square icon-only button used across tables and toolbars. */
export function IconButton({
  icon,
  size = 30,
  iconSize = 15,
  danger = false,
  className,
  ...rest
}: {
  icon: IconName;
  size?: number;
  iconSize?: number;
  danger?: boolean;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      style={{ width: size, height: size }}
      className={cn(
        "flex items-center justify-center rounded-md border-0 bg-transparent text-txt-mute cursor-pointer transition-colors",
        danger ? "hover:bg-red-50 hover:text-red-500" : "hover:bg-panel-2 hover:text-txt",
        className,
      )}
      {...rest}
    >
      <Icon name={icon} size={iconSize} strokeWidth={2.2} />
    </button>
  );
}
