import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { cn } from "./cn";

export interface PopoverProps {
  open: boolean;
  onClose: () => void;
  trigger: ReactNode;
  children: ReactNode;
  align?: "start" | "end";
  title?: string;
  width?: number;
  className?: string;
  panelClassName?: string;
}

const GAP = 4;

/** Always portalled, anchored under the trigger, closes on outside pointer-down or Escape. */
export function Popover({
  open,
  onClose,
  trigger,
  children,
  align = "end",
  title,
  width,
  className,
  panelClassName,
}: PopoverProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ top: number; left: number; width: number } | null>(null);

  useLayoutEffect(() => {
    if (!open) return;
    const node = wrapRef.current;
    if (node === null) return;
    const rect = node.getBoundingClientRect();
    setPosition({ top: rect.bottom + GAP, left: align === "end" ? rect.right : rect.left, width: rect.width });
  }, [open, align]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      const target = event.target as Element;
      if (wrapRef.current?.contains(target) === true) return;
      if (panelRef.current?.contains(target) === true) return;
      // A popover opened from inside this one is portalled elsewhere; clicks in it must not close us.
      if (target.closest("[data-popover]") !== null) return;
      onClose();
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", onClose);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onClose);
    };
  }, [open, onClose]);

  return (
    <div ref={wrapRef} className={cn("relative inline-flex", className)}>
      {trigger}
      {open && position !== null
        ? createPortal(
            <div
              ref={panelRef}
              role="dialog"
              aria-label={title}
              data-popover
              style={{
                top: position.top,
                left: position.left,
                minWidth: width ?? position.width,
                width,
                transform: align === "end" ? "translateX(-100%)" : undefined,
              }}
              className={cn(
                "fixed z-50 flex max-h-[min(520px,80vh)] flex-col rounded-lg border border-line bg-raised shadow-popover",
                panelClassName,
              )}
            >
              {title === undefined ? null : (
                <div className="flex h-8 flex-none items-center border-b border-line px-3 text-xs font-medium text-fg-muted">
                  {title}
                </div>
              )}
              <div className="min-h-0 flex-1 overflow-auto p-1">{children}</div>
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}

export interface MenuItemProps {
  icon?: ReactNode;
  label: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
  danger?: boolean;
  onSelect: () => void;
}

export function MenuItem({ icon, label, hint, disabled = false, danger = false, onSelect }: MenuItemProps) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onSelect}
      className={cn(
        "flex h-7 w-full items-center gap-2 rounded-sm px-2 text-left text-[13px]",
        "disabled:text-fg-disabled",
        danger ? "text-failed hover:bg-failed-bg" : "text-fg hover:bg-hover",
      )}
    >
      {icon === undefined ? null : <span className="shrink-0 text-fg-muted [&>svg]:size-3.5">{icon}</span>}
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {hint === undefined ? null : <span className="shrink-0 text-xs text-fg-muted">{hint}</span>}
    </button>
  );
}

export function MenuSeparator() {
  return <div role="separator" className="my-1 h-px bg-line" />;
}
