import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { cn } from "./cn";
import { Kbd } from "./Surface";

export interface TooltipProps {
  content: ReactNode;
  shortcut?: string;
  side?: "top" | "bottom";
  children: ReactNode;
  className?: string;
}

const SHOW_DELAY = 300;
const GAP = 6;

/** Hover/focus tooltip rendered in a portal so it escapes overflow-hidden panes. */
export function Tooltip({ content, shortcut, side = "bottom", children, className }: TooltipProps) {
  const anchorRef = useRef<HTMLSpanElement>(null);
  const timer = useRef<number | null>(null);
  const [rect, setRect] = useState<DOMRect | null>(null);

  function show() {
    if (timer.current !== null) return;
    timer.current = window.setTimeout(() => {
      timer.current = null;
      const node = anchorRef.current;
      if (node !== null) setRect(node.getBoundingClientRect());
    }, SHOW_DELAY);
  }

  function hide() {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    setRect(null);
  }

  useEffect(() => hide, []);

  const style =
    rect === null
      ? undefined
      : {
          left: rect.left + rect.width / 2,
          top: side === "bottom" ? rect.bottom + GAP : rect.top - GAP,
          transform: side === "bottom" ? "translate(-50%, 0)" : "translate(-50%, -100%)",
        };

  return (
    <span
      ref={anchorRef}
      className={cn("inline-flex", className)}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onMouseDown={hide}
    >
      {children}
      {rect === null
        ? null
        : createPortal(
            <span
              role="tooltip"
              style={style}
              className="pointer-events-none fixed z-[60] flex max-w-96 items-center gap-2 rounded-sm border border-line bg-raised px-2 py-1 text-xs text-fg shadow-popover [&>*]:min-w-0"
            >
              {content}
              {shortcut === undefined ? null : <Kbd>{shortcut}</Kbd>}
            </span>,
            document.body,
          )}
    </span>
  );
}
