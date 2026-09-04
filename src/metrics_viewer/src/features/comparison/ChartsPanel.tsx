import { BarChart3, ChevronsDownUp, ChevronsUpDown, Database, Ellipsis, Eye, RotateCcw } from "lucide-react";
import { lazy, Suspense, useCallback, useMemo, useState } from "react";

import { useRunLineage } from "@/features/lineage/query";
import { isDefaultSql, useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import { useDebouncedCommit } from "@/shared/debounce";
import type { PlotQueryResult, Run, RunScope, XAxis } from "@/shared/types";
import { Button, EmptyState, IconButton, MenuItem, Popover, Range, SearchInput, SegmentedControl, Tooltip } from "@/shared/ui";

import { ChartSection } from "./ChartSection";
import { sectionize, type Plot } from "./logic";
import { QuerySheet } from "./QuerySheet";

const ChartDialog = lazy(() => import("./ChartDialog").then((module) => ({ default: module.ChartDialog })));

const X_AXIS_OPTIONS: { value: XAxis; label: string; title: string; needsTime?: boolean; needsLineage?: boolean }[] = [
  { value: "step", label: "Step", title: "Training step" },
  { value: "lineage", label: "Lineage", title: "Step continued through the runs this one resumed from", needsLineage: true },
  { value: "relative", label: "Time", title: "Seconds since the run started", needsTime: true },
  { value: "wall", label: "Wall", title: "Wall-clock time", needsTime: true },
];

const RUN_SCOPE_OPTIONS: { value: RunScope; label: string; title: string }[] = [
  { value: "selected", label: "Selected", title: "Only the runs ticked in the list" },
  { value: "lineage", label: "+ ancestors", title: "Also the runs the selected ones were resumed from" },
];

interface ChartsPanelProps {
  runs: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
  plots: Plot[];
  result: PlotQueryResult | null;
  error: Error | null;
  fetching: boolean;
  onRefetch: () => void;
  rangeQueries: boolean;
}

export function ChartsPanel({ runs, runColors, chart, plots, result, error, fetching, onRefetch, rangeQueries }: ChartsPanelProps) {
  const viewer = useViewerStore();
  const [filter, setFilter] = useState("");
  const [collapsed, setCollapsed] = useState<string[]>([]);
  const [sqlOpen, setSqlOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [smoothing, setSmoothing] = useDebouncedCommit(
    viewer.globalPlot.smoothing,
    (value) => viewer.setGlobalPlot({ smoothing: value }),
    120,
  );

  const visible = useMemo(
    () =>
      plots.filter(
        (plot) => !viewer.hiddenPlots.includes(plot.name) && plot.name.toLowerCase().includes(filter.toLowerCase()),
      ),
    [plots, viewer.hiddenPlots, filter],
  );
  const sections = useMemo(() => sectionize(visible, viewer.pinnedSections), [visible, viewer.pinnedSections]);
  const hiddenCount = plots.filter((plot) => viewer.hiddenPlots.includes(plot.name)).length;
  const customSql = !isDefaultSql(viewer.sql);
  const dirty = viewer.sql !== viewer.runningSql;
  const hasTime = !rangeQueries && (result === null || result.wall !== null);
  const lineage = useRunLineage();
  const ancestorCount = runs.filter((run) => !viewer.selectedRunIds.includes(run.id)).length;
  const xAxisOptions = useMemo(
    () => X_AXIS_OPTIONS.map((option) => ({
      ...option,
      disabled: (option.needsTime === true && !hasTime) || (option.needsLineage === true && lineage.empty),
    })),
    [hasTime, lineage.empty],
  );

  const runQuery = useCallback(() => {
    if (viewer.sql === viewer.runningSql) onRefetch();
    else viewer.commitSql();
  }, [viewer, onRefetch]);

  function toggleSection(name: string) {
    setCollapsed((current) => (current.includes(name) ? current.filter((item) => item !== name) : [...current, name]));
  }

  const expandedIndex = expanded === null ? -1 : visible.findIndex((plot) => plot.name === expanded);

  return (
    <div className="flex min-h-0 flex-1 flex-row">
      <div className="@container min-h-0 min-w-0 flex-1 overflow-auto">
        <div className="sticky top-0 z-20 flex h-toolbar items-center gap-2 border-b border-line bg-surface px-3">
          <SearchInput label="Filter charts" value={filter} onValue={setFilter} placeholder="Filter charts" className="w-40 @3xl:w-56" />
          <Tooltip content={hasTime ? "What the x axis plots" : "Return wall and rel columns from the query to plot against time"}>
            <SegmentedControl
              label="X axis"
              options={xAxisOptions}
              value={viewer.globalPlot.xAxis}
              onValue={(xAxis) => viewer.setGlobalPlot({ xAxis })}
            />
          </Tooltip>
          <Tooltip
            content={
              lineage.empty
                ? "No checkpoint of this project records the run it was resumed from"
                : "Draw the selected runs, or those plus every run they were resumed from"
            }
          >
            <SegmentedControl
              label="Runs drawn"
              options={RUN_SCOPE_OPTIONS.map((option) => ({
                ...option,
                label: option.value === "lineage" && ancestorCount > 0 ? `${option.label} (${ancestorCount})` : option.label,
                disabled: option.value === "lineage" && lineage.empty,
              }))}
              value={viewer.runScope}
              onValue={viewer.setRunScope}
            />
          </Tooltip>
          <label className="hidden items-center gap-2 text-xs text-fg-muted @2xl:flex">
            Smoothing
            <Range
              aria-label="Smoothing for every chart"
              min={0}
              max={0.99}
              step={0.01}
              value={smoothing}
              onValue={setSmoothing}
              className="w-28"
            />
            <span className="w-8 font-mono tabular-nums text-fg-secondary">{smoothing.toFixed(2)}</span>
          </label>
          <span className="ml-auto flex items-center gap-2">
            <SegmentedControl
              className="hidden @xl:flex"
              label="Chart columns"
              value={viewer.chartColumns}
              onValue={viewer.setChartColumns}
              options={[
                { value: "1" as const, label: "1" },
                { value: "2" as const, label: "2" },
                { value: "3" as const, label: "3" },
                { value: "auto" as const, label: "Auto" },
              ]}
            />
            <Button
              variant={sqlOpen ? "secondary" : "ghost"}
              icon={<Database size={14} />}
              aria-expanded={sqlOpen}
              onClick={() => setSqlOpen(!sqlOpen)}
            >
              SQL
              {error !== null ? (
                <span className="size-1.5 rounded-full bg-failed" aria-label="Query failed" />
              ) : dirty ? (
                <span className="size-1.5 rounded-full bg-queued" aria-label="Edited, not run" />
              ) : customSql ? (
                <span className="size-1.5 rounded-full bg-accent" aria-label="Custom query" />
              ) : null}
            </Button>
            <Popover
              open={menuOpen}
              onClose={() => setMenuOpen(false)}
              width={240}
              trigger={
                <IconButton label="More" active={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>
                  <Ellipsis size={14} />
                </IconButton>
              }
            >
              <MenuItem
                icon={<Eye />}
                label={`Show hidden charts (${hiddenCount})`}
                disabled={hiddenCount === 0}
                onSelect={() => {
                  viewer.showAllPlots();
                  setMenuOpen(false);
                }}
              />
              <MenuItem icon={<ChevronsDownUp />} label="Collapse all sections" onSelect={() => setCollapsed(sections.map((section) => section.name))} />
              <MenuItem icon={<ChevronsUpDown />} label="Expand all sections" onSelect={() => setCollapsed([])} />
              <MenuItem
                icon={<RotateCcw />}
                label="Reset chart settings"
                onSelect={() => {
                  viewer.resetAllPlots();
                  setMenuOpen(false);
                }}
              />
            </Popover>
          </span>
        </div>

        <div className="flex flex-col gap-4 p-4">
          {runs.length === 0 ? (
            <EmptyState icon={<BarChart3 />} title="Select runs to compare" description="Tick runs in the list on the left. Every metric they logged appears here as a chart." />
          ) : null}
          {runs.length > 0 && error !== null ? (
            <EmptyState compact icon={<Database />} title="The query failed" description={error.message}>
              <Button size="sm" onClick={() => setSqlOpen(true)}>Open SQL</Button>
              <Button size="sm" variant="ghost" onClick={viewer.resetSql}>Reset query</Button>
            </EmptyState>
          ) : null}
          {runs.length > 0 && error === null && !fetching && plots.length === 0 && result !== null ? (
            <EmptyState compact icon={<BarChart3 />} title="No metrics for these runs" description="The query returned no rows. Widen it in the SQL panel or pick other runs." />
          ) : null}
          {runs.length > 0 && plots.length > 0 && visible.length === 0 ? (
            <EmptyState compact icon={<BarChart3 />} title="Nothing matches" description={hiddenCount > 0 ? `${hiddenCount} charts are hidden.` : `No chart name contains “${filter}”.`}>
              {hiddenCount > 0 ? <Button size="sm" onClick={viewer.showAllPlots}>Show hidden</Button> : null}
            </EmptyState>
          ) : null}

          {sections.map((section) => (
            <ChartSection
              key={section.name}
              section={section}
              open={!collapsed.includes(section.name)}
              onToggle={() => toggleSection(section.name)}
              columns={viewer.chartColumns}
              runs={runs}
              runColors={runColors}
              chart={chart}
              onExpand={setExpanded}
              rangeQueries={rangeQueries}
            />
          ))}
        </div>
      </div>

      <QuerySheet open={sqlOpen} onClose={() => setSqlOpen(false)} onRun={runQuery} running={fetching} error={error} result={result} plotCount={plots.length} />

      {expandedIndex === -1 ? null : (
        <Suspense fallback={null}>
          <ChartDialog
            plot={visible[expandedIndex]}
            plots={visible}
            runs={runs}
            runColors={runColors}
            chart={chart}
            hasTime={hasTime}
            rangeQuery={rangeQueries}
            onSelectPlot={setExpanded}
            onClose={() => setExpanded(null)}
          />
        </Suspense>
      )}
    </div>
  );
}
