import {
  ArrowLeft,
  Columns2,
  LayoutGrid,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Rows2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Analysis } from "@/features/comparison/Analysis";
import { ModelMonitor } from "@/features/model-monitor/ModelMonitor";
import { Projects } from "@/features/projects/Projects";
import { useProjectsQuery } from "@/features/projects/query";
import { RunTable } from "@/features/runs/RunTable";
import { useRunsQuery } from "@/features/runs/query";
import { Button, GroupLabel, IconButton, SplitPane, StatusBadge } from "@/shared/ui";

import { useViewerLayout } from "./layout";
import { useViewerStore } from "./store";
import { ViewsDialog } from "./ViewsDialog";

const HEADER_HEIGHT = "3.5rem";

export function Viewer() {
  const viewer = useViewerStore();
  const projectsQuery = useProjectsQuery();
  const runsQuery = useRunsQuery(viewer.projectId);
  const { layout, patchLayout, toggleCollapsed } = useViewerLayout();
  const [viewsOpen, setViewsOpen] = useState(false);
  const [modelMonitorOpen, setModelMonitorOpen] = useState(false);
  const runs = runsQuery.data ?? [];

  useEffect(() => {
    viewer.initializeColumns(runs);
  }, [runs, viewer.initializeColumns]);

  const selectedRuns = useMemo(
    () => runs.filter((run) => viewer.selectedRunIds.includes(run.id)),
    [runs, viewer.selectedRunIds],
  );

  if (viewer.projectId === null) {
    return (
      <Projects
        projects={projectsQuery.data ?? []}
        loading={projectsQuery.isPending}
        onOpen={viewer.selectProject}
      />
    );
  }

  const project = projectsQuery.data?.find((candidate) => candidate.id === viewer.projectId);
  const isColumns = layout.orientation === "columns";

  return (
    <main className="flex h-dvh flex-col overflow-hidden">
      <header className="sticky top-0 z-40 flex h-14 flex-none items-center justify-between gap-4 border-b border-line bg-elevated px-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <IconButton
            label="Back to projects"
            onClick={() => modelMonitorOpen ? setModelMonitorOpen(false) : viewer.selectProject(null)}
            variant="secondary"
          >
            <ArrowLeft size={15} />
          </IconButton>
          <div className="flex min-w-0 flex-col">
            <GroupLabel>{modelMonitorOpen ? selectedRuns[0]?.name : "Project"}</GroupLabel>
            <h1 className="m-0 truncate text-sm leading-tight font-semibold tracking-tight text-fg">
              {modelMonitorOpen ? "Model graph" : project?.name ?? "Project"}
            </h1>
          </div>
          {modelMonitorOpen && selectedRuns.length === 1
            ? <StatusBadge status={selectedRuns[0].status} />
            : null}
        </div>

        <div className="flex items-center gap-2">
          {!modelMonitorOpen ? (
            <span className="hidden font-mono text-xs tabular-nums text-fg-muted md:inline">
              {viewer.selectedRunIds.length} / {runs.length} runs
            </span>
          ) : null}
          {!modelMonitorOpen && viewer.selectedRunIds.length > 0 ? (
            <Button
              variant="secondary"
              icon={<X size={13} />}
              onClick={() => viewer.selectRuns([])}
              title={`Deselect all ${viewer.selectedRunIds.length} runs`}
            >
              Clear
            </Button>
          ) : null}
          {!modelMonitorOpen ? (
            <div className="flex items-center gap-0.5 rounded-lg border border-line bg-inset p-0.5">
              <IconButton
                label={layout.collapsed === "start" ? "Show runs table" : "Hide runs table"}
                size="sm"
                active={layout.collapsed === "start"}
                onClick={() => toggleCollapsed("start")}
              >
                {layout.collapsed === "start" ? <PanelLeftOpen size={14} /> : <PanelLeftClose size={14} />}
              </IconButton>
              <IconButton
                label={isColumns ? "Stack panes as rows" : "Place panes side by side"}
                size="sm"
                onClick={() => patchLayout({ orientation: isColumns ? "rows" : "columns" })}
              >
                {isColumns ? <Rows2 size={14} /> : <Columns2 size={14} />}
              </IconButton>
              <IconButton
                label={layout.collapsed === "end" ? "Show analysis" : "Hide analysis"}
                size="sm"
                active={layout.collapsed === "end"}
                onClick={() => toggleCollapsed("end")}
              >
                {layout.collapsed === "end" ? <PanelRightOpen size={14} /> : <PanelRightClose size={14} />}
              </IconButton>
            </div>
          ) : null}
          {!modelMonitorOpen ? (
            <Button
              variant="secondary"
              icon={<LayoutGrid size={14} />}
              aria-expanded={viewsOpen}
              onClick={() => setViewsOpen(true)}
            >
              Views
            </Button>
          ) : null}
        </div>
      </header>

      {modelMonitorOpen && selectedRuns.length === 1 ? (
        <ModelMonitor run={selectedRuns[0]} />
      ) : (
        <SplitPane
          label="Runs and analysis"
          orientation={layout.orientation}
          ratio={layout.ratio}
          onRatio={(ratio) => patchLayout({ ratio })}
          collapsed={layout.collapsed}
          stickyTop={HEADER_HEIGHT}
          onResizeEnd={() => window.dispatchEvent(new Event("resize"))}
          start={
            <RunTable
              runs={runs}
              selected={viewer.selectedRunIds}
              columns={viewer.columns}
              runColors={viewer.runColors}
              starred={viewer.starredRunIds}
              loading={runsQuery.isPending}
              scroll="self"
              onToggle={viewer.toggleRun}
              onSelect={viewer.selectRuns}
              onColumns={viewer.setColumns}
              onRunColor={viewer.setRunColor}
              onStar={viewer.toggleStar}
              className={isColumns ? "border-r border-line" : "border-b border-line"}
            />
          }
          end={
            <Analysis
              runs={selectedRuns}
              onModelGraph={selectedRuns.length === 1 ? () => setModelMonitorOpen(true) : undefined}
            />
          }
        />
      )}

      <ViewsDialog
        open={viewsOpen}
        projectName={project?.name ?? "Project"}
        onClose={() => setViewsOpen(false)}
      />
    </main>
  );
}
