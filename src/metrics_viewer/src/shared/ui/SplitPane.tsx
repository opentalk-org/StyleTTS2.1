import { useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

import { cn } from "./cn";

export type SplitOrientation = "columns" | "rows";
export type SplitCollapsed = "start" | "end" | null;

export interface SplitPaneProps {
  label: string;
  orientation: SplitOrientation;

  ratio: number;
  onRatio: (ratio: number) => void;

  collapsed?: SplitCollapsed;
  start: ReactNode;
  end: ReactNode;
  minRatio?: number;
  maxRatio?: number;





  pageScroll?: boolean;

  stickyTop?: string;

  onResizeEnd?: () => void;
  className?: string;
}

const KEYBOARD_STEP = 0.02;


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

        pageScroll ? "items-stretch" : "overflow-hidden",
        className,
      )}
    >
      {collapsed === "start" ? null : (
        <div
          ref={startRef}
          className={cn(
            "flex min-h-0 min-w-0 flex-col",

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

            pageScroll && isColumns ? "sticky self-start" : "",
          )}
          style={
            pageScroll && isColumns
              ? { top: stickyTop, height: `calc(100dvh - ${stickyTop})` }
              : undefined
          }
        >

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
