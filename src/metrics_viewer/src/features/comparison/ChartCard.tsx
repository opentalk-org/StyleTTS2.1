import { EyeOff, GripVertical, Maximize2, RotateCcw } from "lucide-react";
import { memo, useEffect, useMemo, useRef, useState, type DragEvent, type RefObject } from "react";
import { createPortal } from "react-dom";

import { formatMetric } from "@/features/runs/logic";
import { DEFAULT_PLOT_SETTINGS, useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import { Plot, PLOT_CONFIG, Plotly } from "@/shared/plot";
import type { Run, XAxis } from "@/shared/types";
import { cn, IconButton, Tooltip } from "@/shared/ui";

import { nearestRun, useCursorOverlay, useCursorStore, useRunHighlight } from "@/shared/cursor";
import { usePlotRangeQuery, type PlotRange } from "./query";
import { buildTraces, formatX, groupPlots, plotLayout, resolveSettings, valuesAt, type Plot as PlotData } from "./logic";

const CARD_PLOT_HEIGHT = 200;
const DRAG_TYPE = "application/x-metrics-chart";

interface ChartCardProps {
  plot: PlotData;
  runs: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
  onExpand: (name: string) => void;
  onMove: (from: string, to: string) => void;
  rangeQuery: boolean;
}

export const ChartCard = memo(function ChartCard({ plot, runs, runColors, chart, onExpand, onMove, rangeQuery }: ChartCardProps) {
  const globalPlot = useViewerStore((state) => state.globalPlot);
  const saved = useViewerStore((state) => state.plotSettings[plot.name]);
  const togglePlotHidden = useViewerStore((state) => state.togglePlotHidden);
  const setCursor = useCursorStore((state) => state.set);
  const clearCursor = useCursorStore((state) => state.clear);
  const plotRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const [dropTarget, setDropTarget] = useState(false);
  const graphRef = useRef<HTMLElement | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const rangeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [visible, setVisible] = useState(false);
  const [range, setRange] = useState<PlotRange>({ xMin: null, xMax: null });
  const targetPoints = Math.min(4000, Math.max(200, Math.round((cardRef.current?.clientWidth ?? 680) * 1.5)));
  const rangeResult = usePlotRangeQuery(runs.map((run) => run.id), plot.name, range, targetPoints, rangeQuery && visible);
  const queriedPlot = useMemo(() => groupPlots(rangeResult.data ?? null)[0], [rangeResult.data]);
  const displayedPlot = rangeQuery ? queriedPlot ?? plot : plot;

  const settings = saved ?? DEFAULT_PLOT_SETTINGS;
  const effective = useMemo(() => resolveSettings(settings, globalPlot), [settings, globalPlot]);
  const traces = useMemo(
    () => buildTraces(displayedPlot, runs, effective, runColors, chart),
    [displayedPlot, runs, effective, runColors, chart],
  );
  const layout = useMemo(() => plotLayout(effective, chart, CARD_PLOT_HEIGHT), [effective, chart]);
  useCursorOverlay(graphRef, overlayRef, effective.axis);
  useRunHighlight(graphRef);

  useEffect(() => {
    const card = cardRef.current;
    if (card === null) return;
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: "300px" });
    observer.observe(card);
    return () => observer.disconnect();
  }, []);

  useEffect(() => () => {
    if (rangeTimer.current !== null) clearTimeout(rangeTimer.current);
  }, []);

  function resetZoom() {
    setRange({ xMin: null, xMax: null });
    if (graphRef.current !== null) void Plotly.relayout(graphRef.current, { "xaxis.autorange": true, "yaxis.autorange": true });
  }

  function onRelayout(event: Readonly<Record<string, unknown>>) {
    if (!rangeQuery) return;
    const xMin = Number(event["xaxis.range[0]"]);
    const xMax = Number(event["xaxis.range[1]"]);
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) return;
    if (rangeTimer.current !== null) clearTimeout(rangeTimer.current);
    rangeTimer.current = setTimeout(() => setRange({ xMin, xMax }), 150);
  }

  function onDragOver(event: DragEvent<HTMLDivElement>) {
    if (!event.dataTransfer.types.includes(DRAG_TYPE)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setDropTarget(true);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDropTarget(false);
    onMove(event.dataTransfer.getData(DRAG_TYPE), plot.name);
  }

  return (
    <div
      ref={cardRef}
      onDragOver={onDragOver}
      onDragLeave={() => setDropTarget(false)}
      onDrop={onDrop}
      className={cn(
        "group/card flex min-w-0 flex-col overflow-hidden rounded-md border bg-surface transition-[border-color,box-shadow] duration-100",
        dropTarget ? "border-accent shadow-[0_0_0_2px_var(--color-accent-subtle)]" : "border-line",
      )}
    >
      <div className="flex h-card-head flex-none items-center justify-between gap-2 border-b border-line pr-1 pl-3">
        <Tooltip content={plot.name}>
          <button
            type="button"
            onClick={() => onExpand(plot.name)}
            className="min-w-0 truncate text-left text-[13px] font-medium text-fg hover:text-accent"
          >
            {plot.name}
          </button>
        </Tooltip>
        <div className="flex shrink-0 items-center opacity-0 group-hover/card:opacity-100 focus-within:opacity-100">
          <Tooltip content="Drag to reorder">
            <button
              type="button"
              aria-label="Drag to reorder"
              draggable
              onDragStart={(event) => {
                event.dataTransfer.setData(DRAG_TYPE, plot.name);
                event.dataTransfer.effectAllowed = "move";
                if (cardRef.current !== null) event.dataTransfer.setDragImage(cardRef.current, 20, 16);
              }}
              className="grid size-control-sm cursor-grab place-items-center rounded-md text-fg-muted hover:bg-hover hover:text-fg active:cursor-grabbing"
            >
              <GripVertical size={13} />
            </button>
          </Tooltip>
          <IconButton label="Reset zoom" size="sm" onClick={resetZoom}>
            <RotateCcw size={13} />
          </IconButton>
          <IconButton label="Expand" size="sm" onClick={() => onExpand(plot.name)}>
            <Maximize2 size={13} />
          </IconButton>
          <IconButton label="Hide chart" size="sm" onClick={() => togglePlotHidden(plot.name)}>
            <EyeOff size={13} />
          </IconButton>
        </div>
      </div>
      {/* Explicit height: Plotly's resize handler sizes the plot from this box. */}
      <div ref={plotRef} className="relative bg-plot" style={{ height: CARD_PLOT_HEIGHT }}>
        <Plot
          data={traces}
          layout={layout}
          config={PLOT_CONFIG}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          onInitialized={(_, graph) => {
            graphRef.current = graph;
          }}
          onUpdate={(_, graph) => {
            graphRef.current = graph;
          }}
          onHover={(event) =>
            setCursor({
              x: event.points[0].x as number,
              axis: effective.axis,
              source: plot.name,
              runId: nearestRun(event, graphRef.current),
              pointerY: event.event.clientY,
            })
          }
          onUnhover={clearCursor}
          onRelayout={onRelayout}
        />
        <CursorLine overlayRef={overlayRef} />
      </div>
      <HoverBox plot={displayedPlot} axis={effective.axis} runs={runs} runColors={runColors} anchor={plotRef} />
    </div>
  );
});

/** Dotted vertical line moved by `useCursorOverlay`; hidden until a cursor exists. */
export function CursorLine({ overlayRef }: { overlayRef: RefObject<HTMLDivElement | null> }) {
  return (
    <div
      ref={overlayRef}
      aria-hidden
      style={{ display: "none" }}
      className="pointer-events-none absolute w-px border-l border-dotted border-fg-secondary opacity-60"
    />
  );
}

interface HoverBoxProps {
  plot: PlotData;
  axis: XAxis;
  runs: Run[];
  runColors: Record<string, string>;
  anchor: RefObject<HTMLDivElement | null>;
}

const HOVER_WIDTH = 260;
const HOVER_GAP = 8;

/**
 * Hover values rendered beside the card instead of over the lines. Subscribes to the
 * cursor store itself so the card does not re-render while the pointer moves.
 */
function HoverBox({ plot, axis, runs, runColors, anchor }: HoverBoxProps) {
  const cursorX = useCursorStore((state) => (state.source === plot.name ? state.x : null));
  const pointerY = useCursorStore((state) => (state.source === plot.name ? state.pointerY : 0));
  const highlighted = useCursorStore((state) => state.runId);
  const values = useMemo(() => (cursorX === null ? null : valuesAt(plot, axis, cursorX)), [plot, axis, cursorX]);
  if (cursorX === null || values === null || anchor.current === null) return null;
  const rect = anchor.current.getBoundingClientRect();
  const fitsRight = rect.right + HOVER_GAP + HOVER_WIDTH <= window.innerWidth;
  const left = fitsRight ? rect.right + HOVER_GAP : rect.left - HOVER_GAP - HOVER_WIDTH;
  const rows = runs.filter((run) => values.has(run.id));
  const height = 28 + rows.length * 20 + 8;
  const top = Math.min(Math.max(8, pointerY - height / 2), window.innerHeight - height - 8);
  return createPortal(
    <div
      role="tooltip"
      style={{ left, top, width: HOVER_WIDTH }}
      className="pointer-events-none fixed z-40 rounded-md border border-line bg-raised p-1 shadow-popover"
    >
      <div className="flex h-6 items-center justify-between px-1.5 text-xs">
        <span className="truncate font-medium text-fg">{plot.name}</span>
        <span className="shrink-0 font-mono text-fg-muted">{formatX(axis, cursorX)}</span>
      </div>
      {rows.map((run) => (
        <div key={run.id} className={cn("flex h-5 items-center gap-2 px-1.5 text-xs", highlighted !== null && highlighted !== run.id ? "opacity-50" : "")}>
          <span className="size-2 shrink-0 rounded-full" style={{ background: runColors[run.id] }} />
          <span className={cn("min-w-0 flex-1 truncate", highlighted === run.id ? "font-medium text-fg" : "text-fg-secondary")}>{run.name}</span>
          <span className="shrink-0 font-mono tabular-nums text-fg">{formatMetric(values.get(run.id) as number)}</span>
        </div>
      ))}
    </div>,
    document.body,
  );
}
