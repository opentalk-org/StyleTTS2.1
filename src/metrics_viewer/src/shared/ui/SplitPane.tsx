import { useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

import { cn } from "./cn";

export type SplitOrientation = "columns" | "rows";
export type SplitCollapsed = "start" | "end" | null;

export interface SplitPaneProps {
  label: string;
  orientation: SplitOrientation;
  /** Fraction of the container given to the start pane, 0–1. */
  ratio: number;
  onRatio: (ratio: number) => void;
  /** Which pane is hidden, if any. The remaining pane takes the whole area. */
  collapsed?: SplitCollapsed;
  start: ReactNode;
  end: ReactNode;
  minRatio?: number;
  maxRatio?: number;
  /**
   * The document scrolls instead of each pane. In columns the two panes flow to
   * their natural height side by side; in rows the start pane keeps a viewport
   * fraction of its own and the end pane flows on down the page.
   */
  pageScroll?: boolean;
  /** Sticky offset for the divider in page mode, e.g. an app header height. */
  stickyTop?: string;
  /** Called once a resize gesture finishes, for consumers that must re-measure. */
  onResizeEnd?: () => void;
  className?: string;
}

const KEYBOARD_STEP = 0.02;

/** Two panes with a draggable divider; either pane can be collapsed away. */
export function SplitPane({
  label,
  orientation,
  ratio,
  onRatio,
  collapsed = null,
  start,
  end,
  minRatio = 0.2,
  maxRatio = 0.8,
  pageScroll = false,
  stickyTop = "0px",
  onResizeEnd,
  className,
}: SplitPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const startRef = useRef<HTMLDivElement>(null);
  const isColumns = orientation === "columns";
  // In page mode the start pane is sized against the viewport, not the container,
  // whose height is the whole (scrolling) document.
  const rowsAgainstViewport = pageScroll && !isColumns;

  function clamp(value: number): number {
    return Math.min(maxRatio, Math.max(minRatio, value));
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    const container = containerRef.current;
    if (container === null || event.buttons === 0) return;
    const rect = container.getBoundingClientRect();
    if (isColumns) {
      onRatio(clamp((event.clientX - rect.left) / rect.width));
      return;
    }
    const top = rowsAgainstViewport ? (startRef.current?.getBoundingClientRect().top ?? 0) : rect.top;
    const height = rowsAgainstViewport ? window.innerHeight - top : rect.height;
    onRatio(clamp((event.clientY - top) / height));
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const back = isColumns ? "ArrowLeft" : "ArrowUp";
    const forward = isColumns ? "ArrowRight" : "ArrowDown";
    if (event.key !== back && event.key !== forward) return;
    event.preventDefault();
    onRatio(clamp(ratio + (event.key === back ? -KEYBOARD_STEP : KEYBOARD_STEP)));
    onResizeEnd?.();
  }

  const startSize = isColumns
    ? { width: `${ratio * 100}%`, flex: "0 0 auto" }
    : {
        height: rowsAgainstViewport
          ? `calc((100dvh - ${stickyTop}) * ${ratio})`
          : `${ratio * 100}%`,
        flex: "0 0 auto",
      };

  return (
    <div
      ref={containerRef}
      className={cn(
        "flex min-h-0 min-w-0 flex-1",
        isColumns ? "flex-row" : "flex-col",
        // A scroll container here would trap the page scroll and break sticky.
        pageScroll ? "items-stretch" : "overflow-hidden",
        className,
      )}
    >
      {collapsed === "start" ? null : (
        <div
          ref={startRef}
          className={cn(
            "flex min-h-0 min-w-0 flex-col",
            // Columns flow with the page; a row pane keeps its own scrolling box.
            pageScroll && isColumns ? "" : "overflow-hidden",
          )}
          style={collapsed === "end" ? { flex: "1 1 0%" } : startSize}
        >
          {start}
        </div>
      )}

      {collapsed === null ? (
        <div
          role="separator"
          aria-label={label}
          aria-orientation={isColumns ? "vertical" : "horizontal"}
          aria-valuenow={Math.round(ratio * 100)}
          aria-valuemin={Math.round(minRatio * 100)}
          aria-valuemax={Math.round(maxRatio * 100)}
          tabIndex={0}
          onPointerDown={(event) => event.currentTarget.setPointerCapture(event.pointerId)}
          onPointerMove={onPointerMove}
          onPointerUp={(event) => {
            event.currentTarget.releasePointerCapture(event.pointerId);
            onResizeEnd?.();
          }}
          onKeyDown={onKeyDown}
          onDoubleClick={() => {
            onRatio(0.5);
            onResizeEnd?.();
          }}
          title="Drag to resize, double-click to even out"
          className={cn(
            "group relative z-10 flex-none touch-none bg-transparent",
            isColumns ? "w-1.5 cursor-col-resize" : "h-1.5 cursor-row-resize",
            // The column divider must stay reachable however far the page scrolls.
            pageScroll && isColumns ? "sticky self-start" : "",
          )}
          style={
            pageScroll && isColumns
              ? { top: stickyTop, height: `calc(100dvh - ${stickyTop})` }
              : undefined
          }
        >
          {/* The hairline stays 1px; the grab area around it is what gets wider. */}
          <span
            aria-hidden
            className={cn(
              "absolute bg-line transition-colors duration-150 ease-out",
              "group-hover:bg-accent-border group-focus-visible:bg-accent",
              isColumns ? "inset-y-0 left-1/2 w-px -translate-x-1/2" : "inset-x-0 top-1/2 h-px -translate-y-1/2",
            )}
          />
        </div>
      ) : null}

      {collapsed === "end" ? null : (
        <div
          className={cn("flex min-h-0 min-w-0 flex-1 flex-col", pageScroll ? "" : "overflow-hidden")}
        >
          {end}
        </div>
      )}
    </div>
  );
}
