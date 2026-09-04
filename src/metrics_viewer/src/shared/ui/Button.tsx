import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";
import { Tooltip } from "./Tooltip";

export type ButtonVariant = "primary" | "secondary" | "ghost";
export type ControlSize = "sm" | "md" | "lg";

const BASE =
  "inline-flex items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap " +
  "transition-[background-color,border-color,color] duration-100 ease-out " +
  "disabled:pointer-events-none disabled:text-fg-disabled";

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-accent-fg border border-transparent hover:bg-accent-hover disabled:bg-hover",
  secondary: "bg-surface text-fg border border-strong hover:bg-hover disabled:border-line",
  ghost: "bg-transparent text-fg-secondary border border-transparent hover:bg-hover hover:text-fg",
};

const SIZES: Record<ControlSize, string> = {
  sm: "h-control-sm px-2 text-xs",
  md: "h-control px-2.5 text-[13px]",
  lg: "h-control-lg px-3 text-[13px]",
};

const ICON_SIZES: Record<ControlSize, string> = {
  sm: "size-control-sm",
  md: "size-control",
  lg: "size-control-lg",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ControlSize;
  icon?: ReactNode;
}

export function Button({
  variant = "secondary",
  size = "md",
  icon,
  className,
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button type={type} className={cn(BASE, VARIANTS[variant], SIZES[size], className)} {...rest}>
      {icon}
      {children}
    </button>
  );
}

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  shortcut?: string;
  variant?: ButtonVariant;
  size?: ControlSize;
  active?: boolean;
}

/** Icon-only button. The label is the tooltip and the accessible name. */
export function IconButton({
  label,
  shortcut,
  variant = "ghost",
  size = "md",
  active,
  className,
  children,
  type = "button",
  ...rest
}: IconButtonProps) {
  return (
    <Tooltip content={label} shortcut={shortcut}>
      <button
        type={type}
        aria-label={label}
        aria-pressed={active}
        className={cn(
          BASE,
          "shrink-0 p-0",
          ICON_SIZES[size],
          active === true ? "border border-transparent bg-accent-subtle text-accent" : VARIANTS[variant],
          className,
        )}
        {...rest}
      >
        {children}
      </button>
    </Tooltip>
  );
}
