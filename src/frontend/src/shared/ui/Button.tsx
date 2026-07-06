import type { ButtonHTMLAttributes, ReactNode } from "react";

import { Icon, type IconName } from "../icons";
import { cn } from "./cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-blue-500 text-white hover:bg-blue-600 border-0",
  secondary:
    "bg-panel text-txt border border-line-2 hover:border-txt-mute",
  ghost: "bg-panel-2 text-txt hover:bg-panel-3 border-0",
  danger: "bg-red-500 text-white hover:bg-red-600 border-0",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 px-3 text-xs gap-1.5 rounded-md",
  md: "h-9 px-3.5 text-[13px] gap-1.5 rounded-md",
  lg: "h-11 px-5 text-sm gap-2 rounded-lg",
};

export function Button({
  variant = "secondary",
  size = "md",
  icon,
  children,
  className,
  ...rest
}: {
  variant?: Variant;
  size?: Size;
  icon?: IconName;
  children?: ReactNode;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center font-semibold whitespace-nowrap cursor-pointer transition-colors disabled:cursor-default disabled:opacity-50",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {icon ? <Icon name={icon} size={size === "lg" ? 18 : 15} strokeWidth={2.2} /> : null}
      {children}
    </button>
  );
}
