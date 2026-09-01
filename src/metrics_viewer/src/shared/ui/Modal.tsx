import { useEffect, type ReactNode } from "react";

import { cn } from "./cn";

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  label: string;
  children: ReactNode;
  /** Centers a self-sized dialog instead of filling the viewport height. */
  centered?: boolean;
  className?: string;
}

/** Full-viewport working surface: dimmed canvas, one elevated panel, Escape to close. */
export function Modal({ open, onClose, label, children, centered = false, className }: ModalProps) {
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

  return (
    <div
      className={cn(
        "fixed inset-0 z-50 flex justify-center bg-deep/80 p-4 backdrop-blur-[2px] md:p-6",
        centered ? "items-center" : "items-stretch",
      )}
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={label}
        onClick={(event) => event.stopPropagation()}
        className={cn(
          "flex min-h-0 w-full flex-col overflow-hidden rounded-2xl border border-line bg-elevated shadow-overlay",
          centered ? "max-h-full" : "",
          className,
        )}
      >
        {children}
      </div>
    </div>
  );
}
