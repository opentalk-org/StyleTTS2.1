import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";

export type ButtonVariant = "primary" | "secondary" | "ghost";
export type ControlSize = "sm" | "md";

const BASE =
  "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium whitespace-nowrap " +
  "transition-[background-color,border-color,color,opacity] duration-150 ease-out " +
  "active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40";

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-white border border-transparent hover:bg-accent-bright",
  secondary: "bg-surface text-fg border border-line hover:bg-surface-hover hover:border-line-hover",
  ghost: "bg-transparent text-fg-secondary border border-transparent hover:bg-surface hover:text-fg",
};

const SIZES: Record<ControlSize, string> = {
  sm: "h-7 px-2 text-xs",
  md: "h-8 px-3 text-sm",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ControlSize;
  /** Rendered before the label at the control's icon size. */
  icon?: ReactNode;
}

export function Button({ variant = "secondary", size = "md", icon, className, children, type = "button", ...rest }: ButtonProps) {
  return (
    <button type={type} className={cn(BASE, VARIANTS[variant], SIZES[size], className)} {...rest}>
      {icon}
      {children}
    </button>
  );
}

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Accessible name and native tooltip; icon-only controls always need one. */
  label: string;
  variant?: ButtonVariant;
  size?: ControlSize;
  /** Toggle buttons only: renders the accent selected treatment and exposes aria-pressed. */
  active?: boolean;
}

const ICON_SIZES: Record<ControlSize, string> = { sm: "size-7", md: "size-8" };

export function IconButton({ label, variant = "ghost", size = "md", active, className, children, type = "button", ...rest }: IconButtonProps) {
  return (
    <button
      type={type}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cn(
        BASE,
        "shrink-0 p-0",
        ICON_SIZES[size],
        active === true
          ? "bg-accent-surface text-accent-bright border border-accent-border"
          : VARIANTS[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}
