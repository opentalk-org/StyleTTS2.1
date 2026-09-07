import { PanelLeftOpen, PanelRightOpen } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { withAncestors } from "@/features/lineage/chains";
import { useRunLineage } from "@/features/lineage/query";
import { Panels } from "@/features/comparison/Panels";
import { Projects } from "@/features/projects/Projects";
import { useProjectsQuery } from "@/features/projects/query";
import { RunsPane } from "@/features/runs/RunsPane";
import { useProjectBootstrapQuery, useRunDetailsQuery, useRunMetricsQuery } from "@/features/runs/query";
import { assignRunColors } from "@/shared/chart";
import type { Run } from "@/shared/types";
import { IconButton, SplitPane } from "@/shared/ui";

import { HelpDialog } from "./HelpDialog";
import { useViewerLayout } from "./layout";
import { useViewerStore } from "./store";
import { useTheme } from "./theme";
import { TopBar } from "./TopBar";

export function Viewer() {
  const queryClient = useQueryClient();
  const viewer = useViewerStore();
  const { theme, chart, toggle: toggleTheme } = useTheme();
  const projectsQuery = useProjectsQuery();
  const bootstrapQuery = useProjectBootstrapQuery(viewer.projectId);
  const runDetailsQuery = useRunDetailsQuery(viewer.selectedRunIds);
  const tableMetricNames = useMemo(
    () => viewer.columns.filter((column) => column.startsWith("metric:")).map((column) => column.slice(7)),
    [viewer.columns],
  );
  const runMetricsQuery = useRunMetricsQuery(viewer.projectId, tableMetricNames);
  const { layout, patchLayout, toggleCollapsed, resetLayout } = useViewerLayout();
  const [helpOpen, setHelpOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [revealRunId, setRevealRunId] = useState<string | null>(null);
  const listedRuns = bootstrapQuery.data?.runs ?? [];
  const runDetails = runDetailsQuery.data ?? {};
  const runMetrics = runMetricsQuery.data ?? {};
  const runs = useMemo(
    () => listedRuns.map((run) => {
      const details = runDetails[run.id];
      return details === undefined
        ? { ...run, summary: runMetrics[run.id] ?? {} }
        : { ...run, ...details };
    }),
    [listedRuns, runDetails, runMetrics],
  );

  useEffect(() => {
    if (bootstrapQuery.data !== undefined) viewer.initializeColumns(bootstrapQuery.data.columns.metrics);
  }, [bootstrapQuery.data, viewer.initializeColumns]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target;
      if (target instanceof Element && target.closest("input, textarea, [contenteditable], [role=dialog]") !== null) return;
      if (event.key === "?") {
        event.preventDefault();
        setHelpOpen(true);
      } else if (event.key === "/") {
        event.preventDefault();
        document.querySelector<HTMLInputElement>('[data-shortcut="search"]')?.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  const lineage = useRunLineage();
  // Use case 2: the charts also draw everything the selected runs were resumed from.
  const scopedRunIds = useMemo(
    () => (viewer.runScope === "lineage" ? withAncestors(viewer.selectedRunIds, lineage.chains) : viewer.selectedRunIds),
    [lineage.chains, viewer.runScope, viewer.selectedRunIds],
  );
  const runColors = useMemo(() => {
    const colors = assignRunColors(runs, chart.series, viewer.runColorOverrides);
    // An ancestor that never wrote a status row is missing from the runs list; it still
    // gets drawn, so it still needs a colour of its own.
    let next = runs.length;
    for (const id of scopedRunIds) {
      if (colors[id] !== undefined) continue;
      colors[id] = viewer.runColorOverrides[id] ?? chart.series[next % chart.series.length];
      next += 1;
    }
    return colors;
  }, [runs, chart.series, viewer.runColorOverrides, scopedRunIds]);

  const selectedRuns = useMemo(
    () => runs.filter((run) => viewer.selectedRunIds.includes(run.id)),
    [runs, viewer.selectedRunIds],
  );
  /** Selected runs plus their ancestors, as Run rows the charts can draw. */
  const scopedRuns = useMemo(
    () => scopedRunIds.map((id) => runs.find((run) => run.id === id) ?? placeholderRun(id, lineage.names.get(id))),
    [lineage.names, runs, scopedRunIds],
  );

  const projects = projectsQuery.data ?? [];
  const project = projects.find((candidate) => candidate.id === viewer.projectId);
  const anyRunning = runs.some((run) => run.status === "running");

  async function refreshVisible() {
    setRefreshing(true);
    try {
      await queryClient.refetchQueries({ type: "active" });
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <main className="flex h-dvh flex-col overflow-hidden bg-canvas">
      <TopBar
        projects={projects}
        project={project}
        theme={theme}
        onTheme={toggleTheme}
        onHelp={() => setHelpOpen(true)}
        anyRunning={anyRunning}
        refreshing={refreshing}
        onRefresh={() => void refreshVisible()}
      />

      {viewer.projectId === null ? (
        <Projects projects={projects} loading={projectsQuery.isPending} onOpen={viewer.selectProject} />
      ) : (
        <SplitPane
          label="Runs and panels"
          orientation={layout.orientation}
          size={layout.size}
          onSize={(size) => patchLayout({ size })}
          minSize={layout.orientation === "columns" ? 300 : 200}
          maxSize={layout.orientation === "columns" ? 640 : 600}
          collapsed={layout.collapsed}
          onCollapse={(pane) => patchLayout({ collapsed: pane })}
          onResizeEnd={() => window.dispatchEvent(new Event("resize"))}
          rail={
            <div className="flex w-11 flex-none flex-col items-center gap-2 border-r border-line bg-surface py-2">
              <IconButton label="Show runs" onClick={() => toggleCollapsed("start")}>
                <PanelLeftOpen size={15} />
              </IconButton>
              <span className="rounded-sm bg-accent-subtle px-1 font-mono text-[11px] text-accent">
                {viewer.selectedRunIds.length}
              </span>
            </div>
          }
          endRail={
            <div className="flex w-11 flex-none flex-col items-center border-l border-line bg-surface py-2">
              <IconButton label="Show panels" onClick={() => toggleCollapsed("end")}>
                <PanelRightOpen size={15} />
              </IconButton>
            </div>
          }
          start={
            <RunsPane
              runs={runs}
              projectColumns={bootstrapQuery.data?.columns ?? { params: [], metrics: [] }}
              loading={bootstrapQuery.isPending}
              runColors={runColors}
              palette={chart.series}
              chart={chart}
              layout={layout}
              revealRunId={revealRunId}
              onCollapse={() => toggleCollapsed("start")}
              onStack={() =>
                patchLayout({ orientation: layout.orientation === "columns" ? "rows" : "columns" })
              }
              onResetLayout={resetLayout}
            />
          }
          end={
            <Panels
              runs={selectedRuns}
              scopedRuns={scopedRuns}
              allRuns={runs}
              projectColumns={bootstrapQuery.data?.columns ?? { params: [], metrics: [] }}
              runColors={runColors}
              chart={chart}
              onRevealRun={(runId, additive) => {
                viewer.focusRun(runId, additive);
                setRevealRunId(null);
                requestAnimationFrame(() => setRevealRunId(runId));
                patchLayout({ collapsed: null });
              }}
            />
          }
        />
      )}

      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </main>
  );
}

/** A run the project's run list does not have — drawn from what the lineage knows. */
function placeholderRun(id: string, name: string | undefined): Run {
  return {
    id,
    projectId: "",
    name: name ?? id.slice(0, 8),
    status: "succeeded",
    startedAt: 0,
    endedAt: 0,
    params: {},
    summary: {},
  };
}
