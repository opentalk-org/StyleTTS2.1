import { PanelLeftOpen, PanelRightOpen } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { Panels } from "@/features/comparison/Panels";
import { Projects } from "@/features/projects/Projects";
import { useProjectsQuery } from "@/features/projects/query";
import { RunsPane } from "@/features/runs/RunsPane";
import { useRunDetailsQueries, useRunsQuery } from "@/features/runs/query";
import { assignRunColors } from "@/shared/chart";
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
  const runsQuery = useRunsQuery(viewer.projectId);
  const runDetails = useRunDetailsQueries(viewer.selectedRunIds);
  const { layout, patchLayout, toggleCollapsed, resetLayout } = useViewerLayout();
  const [helpOpen, setHelpOpen] = useState(false);
  const listedRuns = runsQuery.data ?? [];
  const runs = useMemo(
    () => listedRuns.map((run) => ({ ...run, ...runDetails[run.id] })),
    [listedRuns, runDetails],
  );

  useEffect(() => {
    viewer.initializeColumns(runs);
  }, [runs, viewer.initializeColumns]);

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

  const runColors = useMemo(
    () => assignRunColors(runs, chart.series, viewer.runColorOverrides),
    [runs, chart.series, viewer.runColorOverrides],
  );
  const selectedRuns = useMemo(
    () => runs.filter((run) => viewer.selectedRunIds.includes(run.id)),
    [runs, viewer.selectedRunIds],
  );

  const projects = projectsQuery.data ?? [];
  const project = projects.find((candidate) => candidate.id === viewer.projectId);
  const anyRunning = runs.some((run) => run.status === "running");

  function refreshVisible() {
    void runsQuery.refetch();
    void projectsQuery.refetch();
    void queryClient.invalidateQueries({ queryKey: ["run-params"] });
    void queryClient.invalidateQueries({ queryKey: ["run-summary"] });
    void queryClient.invalidateQueries({ queryKey: ["metric-names"] });
    void queryClient.invalidateQueries({ queryKey: ["plot-range"] });
    void queryClient.invalidateQueries({ queryKey: ["plots"] });
    if (viewer.tab === "media") void queryClient.invalidateQueries({ queryKey: ["artifacts"] });
    if (viewer.tab === "graph") {
      void queryClient.invalidateQueries({ queryKey: ["model-graph"] });
      void queryClient.invalidateQueries({ queryKey: ["array-metric-names"] });
      void queryClient.invalidateQueries({ queryKey: ["array-metric"] });
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
        onRefresh={refreshVisible}
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
              loading={runsQuery.isPending}
              runColors={runColors}
              palette={chart.series}
              layout={layout}
              onCollapse={() => toggleCollapsed("start")}
              onStack={() =>
                patchLayout({ orientation: layout.orientation === "columns" ? "rows" : "columns" })
              }
              onResetLayout={resetLayout}
            />
          }
          end={<Panels runs={selectedRuns} allRuns={runs} runColors={runColors} chart={chart} />}
        />
      )}

      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </main>
  );
}
