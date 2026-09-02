import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { cn } from "./cn";

export interface PopoverProps {
  open: boolean;
  onClose: () => void;

  trigger: ReactNode;
  children: ReactNode;
  align?: "start" | "end";




  portal?: boolean;
  className?: string;
  panelClassName?: string;
}


export function Popover({
  open,
  onClose,
  trigger,
  children,
  align = "end",
  portal = false,
  className,
  panelClassName,
}: PopoverProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ top: number; left: number; width: number } | null>(null);

  useLayoutEffect(() => {
    if (!open || !portal) return;
    const trigger = wrapRef.current;
    if (trigger === null) return;
    const rect = trigger.getBoundingClientRect();
    setPosition({
      top: rect.bottom + 6,
      left: align === "end" ? rect.right : rect.left,
      width: rect.width,
    });
  }, [open, portal, align]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      const target = event.target as Node;
      if (wrapRef.current?.contains(target) === true) return;
      if (panelRef.current?.contains(target) === true) return;
      onClose();
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);


    if (portal) {
      document.addEventListener("scroll", onClose, true);
      window.addEventListener("resize", onClose);
    }
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("scroll", onClose, true);
      window.removeEventListener("resize", onClose);
    };
  }, [open, onClose, portal]);

  const panelClasses = cn(
    "z-40 rounded-lg border border-line-hover bg-elevated p-1.5 shadow-overlay",
    panelClassName,
  );

  return (
    <div ref={wrapRef} className={cn("relative", className)}>
      {trigger}
      {open && !portal ? (
        <div
          ref={panelRef}
          className={cn("absolute top-[calc(100%+6px)]", align === "end" ? "right-0" : "left-0", panelClasses)}
        >
          {children}
        </div>
      ) : null}
      {open && portal && position !== null
        ? createPortal(
            <div
              ref={panelRef}
              style={{
                top: position.top,
                left: position.left,
                minWidth: position.width,
                transform: align === "end" ? "translateX(-100%)" : undefined,
              }}
              className={cn("fixed", panelClasses)}
            >
              {children}
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}
