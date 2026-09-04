import { useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

import { cn } from "./cn";

export type SplitOrientation = "columns" | "rows";
export type SplitCollapsed = "start" | "end" | null;

export interface SplitPaneProps {
  label: string;
  orientation: SplitOrientation;
  /** Size of the start pane in px (columns: width, rows: height). */
  size: number;
  onSize: (size: number) => void;
  minSize?: number;
  maxSize?: number;
  collapsed?: SplitCollapsed;
  /** Called when the handle is dragged past a pane's minimum size. */
  onCollapse?: (pane: "start" | "end") => void;
  /** Rendered in place of the start pane while it is collapsed. */
  rail?: ReactNode;
  /** Rendered in place of the end pane while it is collapsed. */
  endRail?: ReactNode;
  start: ReactNode;
  end: ReactNode;
  onResizeEnd?: () => void;
  className?: string;
}

const KEYBOARD_STEP = 16;
/** How far past the minimum size a drag has to go before the pane collapses. */
const COLLAPSE_SLACK = 48;
const MIN_END_SIZE = 240;

export function SplitPane({
  label,
  orientation,
  size,
  onSize,
  minSize = 240,
  maxSize = 720,
  collapsed = null,
  onCollapse,
  rail,
  endRail,
  start,
  end,
  onResizeEnd,
  className,
}: SplitPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isColumns = orientation === "columns";

  function clamp(value: number): number {
    return Math.min(maxSize, Math.max(minSize, value));
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    const container = containerRef.current;
    if (container === null || event.buttons === 0) return;
    const rect = container.getBoundingClientRect();
    const position = isColumns ? event.clientX - rect.left : event.clientY - rect.top;
    const total = isColumns ? rect.width : rect.height;
    if (onCollapse !== undefined && position < minSize - COLLAPSE_SLACK) {
      event.currentTarget.releasePointerCapture(event.pointerId);
      onCollapse("start");
      return;
    }
    if (onCollapse !== undefined && position > total - MIN_END_SIZE + COLLAPSE_SLACK) {
      event.currentTarget.releasePointerCapture(event.pointerId);
      onCollapse("end");
      return;
    }
    onSize(clamp(Math.min(position, total - MIN_END_SIZE)));
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const back = isColumns ? "ArrowLeft" : "ArrowUp";
    const forward = isColumns ? "ArrowRight" : "ArrowDown";
    if (event.key !== back && event.key !== forward) return;
    event.preventDefault();
    onSize(clamp(size + (event.key === back ? -KEYBOARD_STEP : KEYBOARD_STEP)));
    onResizeEnd?.();
  }

  const startStyle = isColumns ? { width: size } : { height: size };

  return (
    <div
      ref={containerRef}
      className={cn("flex min-h-0 min-w-0 flex-1 overflow-hidden", isColumns ? "flex-row" : "flex-col", className)}
    >
      {collapsed === "start" ? (
        rail
      ) : (
        <div
          className="flex min-h-0 min-w-0 flex-none flex-col overflow-hidden"
          style={collapsed === "end" ? { flex: "1 1 0%" } : startStyle}
        >
          {start}
        </div>
      )}

      {collapsed === null ? (
        <div
          role="separator"
          aria-label={label}
          aria-orientation={isColumns ? "vertical" : "horizontal"}
          aria-valuenow={Math.round(size)}
          aria-valuemin={minSize}
          aria-valuemax={maxSize}
          tabIndex={0}
          onPointerDown={(event) => event.currentTarget.setPointerCapture(event.pointerId)}
          onPointerMove={onPointerMove}
          onPointerUp={(event) => {
            event.currentTarget.releasePointerCapture(event.pointerId);
            onResizeEnd?.();
          }}
          onKeyDown={onKeyDown}
          title="Drag to resize"
          className={cn(
            "group relative z-10 flex-none touch-none",
            isColumns ? "w-1.5 cursor-col-resize" : "h-1.5 cursor-row-resize",
          )}
        >
          <span
            aria-hidden
            className={cn(
              "absolute bg-line transition-colors duration-100 ease-out group-hover:bg-accent group-focus-visible:bg-accent",
              isColumns ? "inset-y-0 left-0 w-px" : "inset-x-0 top-0 h-px",
            )}
          />
        </div>
      ) : null}

      {collapsed === "end" ? endRail : (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{end}</div>
      )}
    </div>
  );
}
