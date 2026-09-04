import { X } from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { cn } from "./cn";
import { IconButton } from "./Button";

export type DialogSize = "sm" | "md" | "lg" | "full";

export interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  eyebrow?: ReactNode;
  size?: DialogSize;
  actions?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
  className?: string;
}

const SIZES: Record<DialogSize, string> = {
  sm: "max-w-[420px]",
  md: "max-w-[640px]",
  lg: "max-w-[1100px]",
  full: "h-full max-w-none",
};

export function Dialog({
  open,
  onClose,
  title,
  eyebrow,
  size = "md",
  actions,
  footer,
  children,
  className,
}: DialogProps) {
  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-scrim p-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        onClick={(event) => event.stopPropagation()}
        className={cn(
          "flex max-h-full w-full min-h-0 flex-col overflow-hidden rounded-lg border border-line bg-raised shadow-dialog",
          SIZES[size],
          className,
        )}
      >
        <header className="flex h-11 flex-none items-center justify-between gap-3 border-b border-line pr-2 pl-4">
          <div className="flex min-w-0 items-baseline gap-2">
            {eyebrow === undefined ? null : <span className="shrink-0 text-xs text-fg-muted">{eyebrow}</span>}
            <h2 className="truncate text-sm font-semibold text-fg">{title}</h2>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {actions}
            <IconButton label="Close" shortcut="Esc" onClick={onClose}>
              <X size={14} />
            </IconButton>
          </div>
        </header>
        <div className="flex min-h-0 flex-1 flex-col overflow-auto">{children}</div>
        {footer === undefined ? null : (
          <footer className="flex h-12 flex-none items-center justify-end gap-2 border-t border-line px-3">
            {footer}
          </footer>
        )}
      </div>
    </div>,
    document.body,
  );
}
