import { ChevronLeft, ChevronRight, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { DEFAULT_PLOT_SETTINGS, useViewerStore } from "@/features/viewer/store";
import { formatMetric } from "@/features/runs/logic";
import type { ChartTheme } from "@/shared/chart";
import { EXPANDED_PLOT_CONFIG, Plot, Plotly } from "@/shared/plot";
import type { Run, XAxis } from "@/shared/types";
import { Caption, cn, Dialog, IconButton } from "@/shared/ui";

import { CursorLine, HoverBox } from "./ChartCard";

import { ChartSettingsForm } from "./ChartSettings";
import { nearestRun, pointerDataX, useCursorOverlay, useCursorStore, useRunHighlight } from "@/shared/cursor";
import { buildTraces, clipAncestors, formatX, groupPlots, plotLayout, resolveSettings, seriesStats, valuesAt, type Plot as PlotData } from "./logic";
import { useAncestorCuts, useAncestorRunIds, useLineageGroups, useRunLineage } from "@/features/lineage/query";

import { usePlotRangeQuery, type PlotRange } from "./query";

interface ChartDialogProps {
  plot: PlotData;
  plots: PlotData[];
  runs: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
  hasTime: boolean;
  rangeQuery: boolean;
  onSelectPlot: (name: string) => void;
  onClose: () => void;
}

const LEGEND_COLUMNS = "grid-cols-[12px_1fr_72px_64px_64px]";

export function ChartDialog({ plot, plots, runs, runColors, chart, hasTime, rangeQuery, onSelectPlot, onClose }: ChartDialogProps) {
  const globalPlot = useViewerStore((state) => state.globalPlot);
  const saved = useViewerStore((state) => state.plotSettings[plot.name]);
  const updatePlot = useViewerStore((state) => state.updatePlot);
  const resetPlot = useViewerStore((state) => state.resetPlot);
  const [cursorX, setCursorX] = useState<number | null>(null);
  const graphRef = useRef<HTMLElement | null>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const rangeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [range, setRange] = useState<PlotRange>({ xMin: null, xMax: null });
  const lineage = useRunLineage();
  const ancestors = useAncestorRunIds();
  const groups = useLineageGroups(runs.map((run) => run.id));
  // A parent that kept training past the fork is history only up to it.
  const cuts = useAncestorCuts(runs.map((run) => run.id));
  const settings = saved ?? DEFAULT_PLOT_SETTINGS;
  const effective = useMemo(() => resolveSettings(settings, globalPlot), [settings, globalPlot]);
  // A window picked on one axis means nothing on another.
  useEffect(() => setRange({ xMin: null, xMax: null }), [effective.axis]);
  const rangeResult = usePlotRangeQuery(runs.map((run) => run.id), plot.name, queryRange(range, effective.axis, runs, lineage.offsets), 3000, rangeQuery);
  const queriedPlot = useMemo(() => groupPlots(rangeResult.data ?? null, lineage.offsets)[0], [lineage.offsets, rangeResult.data]);
  const displayedPlot = useMemo(
    () => clipAncestors(rangeQuery ? queriedPlot ?? plot : plot, cuts) as PlotData,
    [cuts, plot, queriedPlot, rangeQuery],
  );

  const traces = useMemo(
    () => buildTraces(displayedPlot, runs, effective, runColors, chart, ancestors),
    [ancestors, displayedPlot, runs, effective, runColors, chart],
  );
  const layout = useMemo(() => plotLayout(effective, chart, displayedPlot), [effective, chart, displayedPlot]);
  useCursorOverlay(graphRef, overlayRef, effective.axis);
  useRunHighlight(graphRef);
  const setCursor = useCursorStore((state) => state.set);
  const clearCursor = useCursorStore((state) => state.clear);
  const highlighted = useCursorStore((state) => state.runId);
  const highlightedGroup = useCursorStore((state) => state.runIds);
  const lit = (runId: string) => highlighted === null || (highlightedGroup ?? new Set([highlighted])).has(runId);
  const cursorValues = useMemo(() => valuesAt(displayedPlot, effective, cursorX), [displayedPlot, effective, cursorX]);

  const index = plots.findIndex((candidate) => candidate.name === plot.name);
  const previous = index > 0 ? plots[index - 1].name : null;
  const next = index < plots.length - 1 ? plots[index + 1].name : null;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.target as HTMLElement).closest("input") !== null) return;
      if (event.key === "ArrowLeft" && previous !== null) onSelectPlot(previous);
      if (event.key === "ArrowRight" && next !== null) onSelectPlot(next);
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [previous, next, onSelectPlot]);

  useEffect(() => () => {
    if (rangeTimer.current !== null) clearTimeout(rangeTimer.current);
  }, []);

  useEffect(() => setRange({ xMin: null, xMax: null }), [plot.name]);

  function onRelayout(event: Readonly<Record<string, unknown>>) {
    if (!rangeQuery) return;
    const xMin = Number(event["xaxis.range[0]"]);
    const xMax = Number(event["xaxis.range[1]"]);
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) return;
    if (rangeTimer.current !== null) clearTimeout(rangeTimer.current);
    rangeTimer.current = setTimeout(() => setRange({ xMin, xMax }), 150);
  }

  return (
    <Dialog
      open
      onClose={onClose}
      title={plot.name}
      size="full"
      actions={
        <>
          <IconButton
            label="Reset zoom"
            onClick={() => {
              setRange({ xMin: null, xMax: null });
              if (graphRef.current !== null) void Plotly.relayout(graphRef.current, { "xaxis.autorange": true, "yaxis.autorange": true });
            }}
          >
            <RotateCcw size={14} />
          </IconButton>
          <IconButton label="Previous chart" shortcut="←" disabled={previous === null} onClick={() => previous !== null && onSelectPlot(previous)}>
            <ChevronLeft size={14} />
          </IconButton>
          <IconButton label="Next chart" shortcut="→" disabled={next === null} onClick={() => next !== null && onSelectPlot(next)}>
            <ChevronRight size={14} />
          </IconButton>
        </>
      }
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1fr_380px]">
        <div ref={plotRef} className="relative min-h-0 bg-plot p-2">
          <Plot
            data={traces}
            layout={layout}
            config={EXPANDED_PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
            onInitialized={(_, graph) => { graphRef.current = graph; }}
            onUpdate={(_, graph) => { graphRef.current = graph; }}
            onHover={(event) => {
              const x = pointerDataX(event, graphRef.current);
              setCursorX(x);
              const runId = nearestRun(event, graphRef.current);
              setCursor({
                x,
                axis: effective.axis,
                source: `dialog:${plot.name}`,
                runId,
                runIds: runId === null ? null : groups.get(runId) ?? null,
                pointerX: event.event.clientX,
                pointerY: event.event.clientY,
              });
            }}
            onUnhover={() => {
              setCursorX(null);
              clearCursor();
            }}
            onRelayout={onRelayout}
          />
          <CursorLine overlayRef={overlayRef} />
          <HoverBox
            plot={displayedPlot}
            settings={effective}
            runs={runs}
            runColors={runColors}
            anchor={plotRef}
            source={`dialog:${plot.name}`}
            besidePointer
          />
        </div>
        <div className="flex min-h-0 flex-col border-l border-line">
          <div className="max-h-[40%] overflow-auto border-b border-line">
            <div className={cn("sticky top-0 grid items-center gap-2 bg-raised px-3 py-1.5", LEGEND_COLUMNS)}>
              <span />
              <Caption>Run</Caption>
              <Caption className="truncate text-right">{cursorX === null ? "Last" : formatX(effective.axis, cursorX)}</Caption>
              <Caption className="text-right">Min</Caption>
              <Caption className="text-right">Max</Caption>
            </div>
            {runs.map((run) => {
              const series = displayedPlot.series.find((item) => item.runId === run.id);
              const stats = series === undefined ? null : seriesStats(series, effective.axis);
              return (
                <div key={run.id} className={cn("grid items-center gap-2 px-3 py-1 text-xs", LEGEND_COLUMNS, highlighted === run.id ? "bg-hover" : "", lit(run.id) ? "" : "opacity-50")}>
                  <span className="size-2.5 rounded-full" style={{ background: runColors[run.id] }} />
                  <span className="truncate text-fg" title={run.name}>{run.name}</span>
                  <span className="text-right font-mono tabular-nums text-fg">{stats === null ? "—" : formatMetric(cursorValues.get(run.id) ?? stats.last)}</span>
                  <span className="text-right font-mono tabular-nums text-fg-secondary" title={stats === null ? undefined : `at ${formatX(effective.axis, stats.minX)}`}>{stats === null ? "—" : formatMetric(stats.min)}</span>
                  <span className="text-right font-mono tabular-nums text-fg-secondary" title={stats === null ? undefined : `at ${formatX(effective.axis, stats.maxX)}`}>{stats === null ? "—" : formatMetric(stats.max)}</span>
                </div>
              );
            })}
          </div>
          <div className="min-h-0 flex-1 overflow-auto p-1">
            <ChartSettingsForm
              settings={settings}
              global={globalPlot}
              onChange={(patch) => updatePlot(plot.name, patch)}
              onReset={saved === undefined ? undefined : () => resetPlot(plot.name)}
              metrics={plots.map((candidate) => candidate.name)}
              metric={plot.name}
              onMetric={onSelectPlot}
              hasTime={hasTime}
            />
          </div>
        </div>
      </div>
    </Dialog>
  );
}

/**
 * The zoom window in query space. On the lineage axis every run is shifted, so the window
 * is widened to cover the shift of every run drawn — the server filters on raw steps.
 */
function queryRange(range: PlotRange, axis: XAxis, runs: Run[], offsets: Map<string, number>): PlotRange {
  if (axis !== "lineage" || range.xMin === null || range.xMax === null || runs.length === 0) return range;
  const shifts = runs.map((run) => offsets.get(run.id) ?? 0);
  return { xMin: range.xMin - Math.max(...shifts), xMax: range.xMax - Math.min(...shifts) };
}
