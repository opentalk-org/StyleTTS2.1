import { BarChart3, ChevronDown, ChevronRight, Columns3, Eye, Loader2, SlidersHorizontal } from "lucide-react";
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import type { Artifact, PlotQueryResult, PlotSettings, Run } from "@/shared/types";
import {
  Button,
  cn,
  CountPill,
  EmptyState,
  SearchInput,
  SegmentedControl,
} from "@/shared/ui";
import { DEFAULT_PLOT_SETTINGS, DEFAULT_SQL, useViewerStore } from "@/state/store";
import { groupPlots, type Plot } from "./logic";
import { MediaPanel } from "./MediaPanel";
import { ParamsPanel } from "./ParamsPanel";
import { QueryBar } from "./QueryBar";
import { useArtifactsQuery, usePlotsQuery } from "./query";

type Tab = "plots" | "params";

const TABS: { id: Tab; label: string; icon: ReactNode }[] = [
  { id: "plots", label: "Plots", icon: <BarChart3 size={14} /> },
  { id: "params", label: "Parameters", icon: <SlidersHorizontal size={14} /> },
];

const MetricChart = lazy(() =>
  import("./MetricChart").then((module) => ({ default: module.MetricChart })),
);

export function Analysis({ runs }: { runs: Run[] }) {
  const [tab, setTab] = useState<Tab>("plots");
  const [metricQuery, setMetricQuery] = useState("");
  const [chartColumns, setChartColumns] = useState<1 | 2 | 3>(2);
  const viewer = useViewerStore();
  const artifactsQuery = useArtifactsQuery(runs);
  const plotsQuery = usePlotsQuery(viewer.projectId, viewer.runningSql, viewer.selectedRunIds);

  const { refetch } = plotsQuery;
  const { sql, runningSql, commitSql } = viewer;

  const runQuery = useCallback(() => {
    if (sql === runningSql) void refetch();
    else commitSql();
  }, [sql, runningSql, commitSql, refetch]);

  return (
    <section aria-label="Analysis" className="flex min-h-0 min-w-0 flex-1 flex-col bg-base">
      <nav
        className="sticky top-14 z-20 flex h-13 flex-none items-center gap-1 border-b border-line bg-base px-3"
        aria-label="Analysis views"
      >
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-current={tab === item.id ? "page" : undefined}
            onClick={() => setTab(item.id)}
            className={cn(
              "flex h-8 items-center gap-2 rounded-lg px-3 text-sm font-medium",
              "transition-[background-color,color,box-shadow] duration-150 ease-out",
              tab === item.id
                ? "bg-accent-surface text-accent-bright shadow-[inset_0_0_0_1px_rgb(99_102_241/0.18)]"
                : "text-fg-muted hover:bg-surface hover:text-fg-secondary",
            )}
          >
            {item.icon}
            {item.label}
          </button>
        ))}
        <span className="ml-auto flex items-center gap-2 font-mono text-xs tabular-nums text-fg-muted">
          {plotsQuery.isFetching || artifactsQuery.isFetching ? (
            <span className="flex items-center gap-1.5 text-accent-bright">
              <Loader2 size={12} className="animate-spin motion-reduce:animate-none" />
              loading
            </span>
          ) : null}
          {runs.length} selected
        </span>
      </nav>

      <div className="@container min-h-0 flex-1 p-4">
        {tab === "plots" ? (
          <Plots
            runs={runs}
            result={plotsQuery.data ?? null}
            error={plotsQuery.error}
            running={plotsQuery.isFetching}
            onRun={runQuery}
            artifacts={artifactsQuery.data ?? []}
            metricQuery={metricQuery}
            onMetricQuery={setMetricQuery}
            columns={chartColumns}
            onColumns={setChartColumns}
          />
        ) : (
          <ParamsPanel runs={runs} />
        )}
      </div>
    </section>
  );
}

interface PlotsProps {
  runs: Run[];
  result: PlotQueryResult | null;
  error: Error | null;
  running: boolean;
  onRun: () => void;
  artifacts: Artifact[];
  metricQuery: string;
  onMetricQuery: (value: string) => void;
  columns: 1 | 2 | 3;
  onColumns: (value: 1 | 2 | 3) => void;
}



const COLUMN_CLASSES: Record<1 | 2 | 3, string> = {
  1: "grid-cols-1",
  2: "grid-cols-1 @3xl:grid-cols-2",
  3: "grid-cols-1 @3xl:grid-cols-2 @6xl:grid-cols-3",
};

function Plots({
  runs,
  result,
  error,
  running,
  onRun,
  artifacts,
  metricQuery,
  onMetricQuery,
  columns,
  onColumns,
}: PlotsProps) {
  const viewer = useViewerStore();
  const [cursorIndex, setCursorIndex] = useState<number | null>(null);
  const onCursorIndex = useThrottledCursor(setCursorIndex);

  const plots = useMemo(() => groupPlots(result), [result]);
  const visible = useMemo(
    () =>
      plots.filter(
        (plot) =>
          !viewer.hiddenPlots.includes(plot.name) &&
          plot.name.toLowerCase().includes(metricQuery.toLowerCase()),
      ),
    [plots, viewer.hiddenPlots, metricQuery],
  );
  const groups = useMemo(() => groupByNamespace(visible, artifacts, metricQuery), [
    visible,
    artifacts,
    metricQuery,
  ]);
  const hiddenCount = plots.length - visible.length - countFilteredOut(plots, metricQuery);

  return (
    <div className="flex flex-col gap-4">
      <QueryBar
        sql={viewer.sql}
        onSql={viewer.setSql}
        onRun={onRun}
        running={running}
        error={error}
        summary={
          result === null
            ? null
            : `${plots.length} plots · ${result.x.length.toLocaleString()} points · ${result.elapsedMs} ms`
        }
        dirty={viewer.sql !== viewer.runningSql}
        onReset={() => {
          viewer.setSql(DEFAULT_SQL);
          viewer.commitSql();
          viewer.showAllPlots();
        }}
      />

      {runs.length === 0 ? (
        <EmptyState
          icon={<BarChart3 />}
          title="No runs selected"
          description="Pick runs on the left. The query above resolves selected() against them and draws one chart per plot column it returns."
        />
      ) : (
        <>
          <div className="flex items-center justify-between gap-3">
            <SearchInput
              label="Filter plots"
              value={metricQuery}
              onValue={onMetricQuery}
              placeholder="Filter plots"
              className="w-72"
            />
            <div className="flex items-center gap-2">
              {hiddenCount > 0 ? (
                <Button variant="secondary" icon={<Eye size={14} />} onClick={viewer.showAllPlots}>
                  Show {hiddenCount} hidden
                </Button>
              ) : null}
              <SegmentedControl
                label="Chart columns"
                leading={<Columns3 size={14} />}
                value={columns}
                onValue={(value) => onColumns(value)}
                options={[
                  { value: 1 as const, label: "1", title: "One column" },
                  { value: 2 as const, label: "2", title: "Two columns" },
                  { value: 3 as const, label: "3", title: "Three columns" },
                ]}
              />
            </div>
          </div>

          {groups.length === 0 ? (
            <EmptyState
              compact
              icon={<BarChart3 />}
              title={plots.length === 0 ? "The query returned no plots" : "Nothing matches this filter"}
              description={
                plots.length === 0
                  ? "Run the query above, or widen its names => [...] list."
                  : `No plot or artifact name contains “${metricQuery}”.`
              }
            />
          ) : null}

          <Suspense fallback={null}>
            {groups.map(([namespace, content]) => (
              <PlotGroup
                key={namespace}
                name={namespace}
                count={content.plots.length + artifactNameCount(content.artifacts)}
              >
                {content.plots.length === 0 ? null : (
                  <div className={cn("grid gap-3", COLUMN_CLASSES[columns])}>
                    {content.plots.map((plot) => (
                      <MetricChart
                        key={plot.name}
                        plot={plot}
                        runs={runs}
                        settings={settingsFor(viewer.plotSettings, plot.name)}
                        runColors={viewer.runColors}
                        cursorIndex={cursorIndex}
                        onCursorIndex={onCursorIndex}
                        onChange={(patch) => viewer.updatePlot(plot.name, patch)}
                        onReset={() => viewer.resetPlot(plot.name)}
                        onHide={() => viewer.togglePlotHidden(plot.name)}
                      />
                    ))}
                  </div>
                )}
                <MediaPanel runs={runs} artifacts={content.artifacts} />
              </PlotGroup>
            ))}
          </Suspense>
        </>
      )}
    </div>
  );
}

function settingsFor(saved: Record<string, PlotSettings>, plot: string): PlotSettings {
  return saved[plot] ?? DEFAULT_PLOT_SETTINGS;
}

function PlotGroup({ name, count, children }: { name: string; count: number; children: ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <section className="flex flex-col gap-3">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex h-9 w-full items-center gap-2 border-b border-line px-1 text-left text-fg-muted transition-colors duration-150 hover:text-fg-secondary"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <strong className="text-sm font-semibold tracking-tight text-fg capitalize">{name}</strong>
        <CountPill>{count}</CountPill>
      </button>
      {open ? children : null}
    </section>
  );
}





function useThrottledCursor(setCursor: (index: number | null) => void) {
  const frame = useRef<number | null>(null);
  const pending = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
    },
    [],
  );

  return useCallback(
    (index: number | null) => {
      pending.current = index;
      if (frame.current !== null) return;
      frame.current = requestAnimationFrame(() => {
        frame.current = null;
        setCursor(pending.current);
      });
    },
    [setCursor],
  );
}

interface GroupContent {
  plots: Plot[];
  artifacts: Artifact[];
}


function groupByNamespace(
  plots: Plot[],
  artifacts: Artifact[],
  query: string,
): [string, GroupContent][] {
  const groups = new Map<string, GroupContent>();
  for (const plot of plots) {
    const key = namespaceOf(plot.name);
    const content = groups.get(key) ?? { plots: [], artifacts: [] };
    content.plots.push(plot);
    groups.set(key, content);
  }
  const normalized = query.toLowerCase();
  for (const artifact of artifacts) {
    if (!artifact.name.toLowerCase().includes(normalized)) continue;
    const key = namespaceOf(artifact.name);
    const content = groups.get(key) ?? { plots: [], artifacts: [] };
    content.artifacts.push(artifact);
    groups.set(key, content);
  }
  return [...groups.entries()];
}

function namespaceOf(name: string): string {
  const separator = name.indexOf("/");
  return separator === -1 ? "other" : name.slice(0, separator);
}

function countFilteredOut(plots: Plot[], query: string): number {
  const normalized = query.toLowerCase();
  return plots.filter((plot) => !plot.name.toLowerCase().includes(normalized)).length;
}

function artifactNameCount(artifacts: Artifact[]): number {
  return new Set(artifacts.map((artifact) => artifact.name)).size;
}
