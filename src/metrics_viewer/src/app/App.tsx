import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  ArrowLeft,
  BookmarkPlus,
  Columns2,
  LayoutGrid,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Rows2,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { Analysis } from "@/features/comparison/Analysis";
import { ModelMonitor } from "@/features/model-monitor/ModelMonitor";
import { Projects } from "@/features/projects/Projects";
import { useProjectsQuery } from "@/features/projects/query";
import { RunTable } from "@/features/runs/RunTable";
import { useRunsQuery } from "@/features/runs/query";
import type { Workspace } from "@/shared/types";
import {
  Button,
  GroupLabel,
  IconButton,
  Modal,
  SplitPane,
  StatusBadge,
  type SplitCollapsed,
  type SplitOrientation,
} from "@/shared/ui";
import { useViewerStore } from "@/state/store";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <Viewer />
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

const LAYOUT_KEY = "runflow.metrics.layout.v1";

const HEADER_HEIGHT = "3.5rem";

interface Layout {
  orientation: SplitOrientation;
  ratio: number;
  collapsed: SplitCollapsed;
}

const DEFAULT_LAYOUT: Layout = { orientation: "columns", ratio: 0.38, collapsed: null };

function loadLayout(): Layout {
  try {
    const stored = localStorage.getItem(LAYOUT_KEY);
    return stored === null ? DEFAULT_LAYOUT : { ...DEFAULT_LAYOUT, ...(JSON.parse(stored) as Layout) };
  } catch {
    return DEFAULT_LAYOUT;
  }
}

function Viewer() {
  const viewer = useViewerStore();
  const projectsQuery = useProjectsQuery();
  const runsQuery = useRunsQuery(viewer.projectId);
  const [layout, setLayout] = useState<Layout>(loadLayout);
  const [viewsOpen, setViewsOpen] = useState(false);
  const [modelMonitorOpen, setModelMonitorOpen] = useState(false);
  const runs = runsQuery.data ?? [];
  const selectedRuns = useMemo(
    () => runs.filter((run) => viewer.selectedRunIds.includes(run.id)),
    [runs, viewer.selectedRunIds],
  );

  useEffect(() => {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  }, [layout]);


  useEffect(() => {
    const frame = requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    return () => cancelAnimationFrame(frame);
  }, [layout.orientation, layout.collapsed]);

  function patchLayout(patch: Partial<Layout>) {
    setLayout((current) => ({ ...current, ...patch }));
  }

  function toggleCollapsed(pane: "start" | "end") {
    patchLayout({ collapsed: layout.collapsed === pane ? null : pane });
  }

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
          <IconButton label="Back to projects" onClick={() => modelMonitorOpen ? setModelMonitorOpen(false) : viewer.selectProject(null)} variant="secondary">
            <ArrowLeft size={15} />
          </IconButton>
          <div className="flex min-w-0 flex-col">
            <GroupLabel>{modelMonitorOpen ? selectedRuns[0]?.name : "Project"}</GroupLabel>
            <h1 className="m-0 truncate text-sm leading-tight font-semibold tracking-tight text-fg">
              {modelMonitorOpen ? "Model graph" : project?.name ?? "Project"}
            </h1>
          </div>
          {modelMonitorOpen && selectedRuns.length === 1 ? (
            <StatusBadge status={selectedRuns[0].status} />
          ) : null}
        </div>

        <div className="flex items-center gap-2">
          {!modelMonitorOpen ? <span className="hidden font-mono text-xs tabular-nums text-fg-muted md:inline">
            {viewer.selectedRunIds.length} / {runs.length} runs
          </span> : null}
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
          {!modelMonitorOpen ? <div className="flex items-center gap-0.5 rounded-lg border border-line bg-inset p-0.5">
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
          </div> : null}
          {!modelMonitorOpen ? <Button
            variant="secondary"
            icon={<LayoutGrid size={14} />}
            aria-expanded={viewsOpen}
            onClick={() => setViewsOpen(true)}
          >
            Views
          </Button> : null}
        </div>
      </header>

      {modelMonitorOpen && selectedRuns.length === 1 ? (
        <ModelMonitor run={selectedRuns[0]} />
      ) : <SplitPane
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
        end={<Analysis
          runs={selectedRuns}
          onModelGraph={selectedRuns.length === 1 ? () => setModelMonitorOpen(true) : undefined}
        />}
      />}

      <ViewsDialog
        open={viewsOpen}
        projectName={project?.name ?? "Project"}
        onClose={() => setViewsOpen(false)}
      />
    </main>
  );
}

function ViewsDialog({
  open,
  projectName,
  onClose,
}: {
  open: boolean;
  projectName: string;
  onClose: () => void;
}) {
  const { workspaces, loadWorkspace, saveWorkspace, deleteWorkspace } = useViewerStore();

  function saveCurrentView() {
    const name = window.prompt("Name this view", `${projectName} comparison`);
    if (name !== null && name.length > 0) saveWorkspace(name);
  }

  function removeView(workspace: Workspace) {
    if (!window.confirm(`Delete the view “${workspace.name}”?`)) return;
    deleteWorkspace(workspace.id);
  }

  return (
    <Modal open={open} onClose={onClose} label="Views" centered className="max-w-2xl">
      <header className="flex h-14 flex-none items-center justify-between gap-3 border-b border-line px-4">
        <div className="flex min-w-0 flex-col gap-1">
          <GroupLabel>Saved configurations</GroupLabel>
          <h2 className="m-0 text-base leading-tight font-semibold tracking-tight text-fg">Views</h2>
        </div>
        <IconButton label="Close views" onClick={onClose}>
          <X size={15} />
        </IconButton>
      </header>

      <div className="min-h-0 flex-1 overflow-auto">
        {workspaces.length === 0 ? (
          <p className="m-0 px-4 py-10 text-center text-xs leading-relaxed text-fg-muted">
            A view stores the plots, table columns, run colors and query you have set up.
            <br />
            Save one to come back to this exact setup later.
          </p>
        ) : (
          workspaces.map((workspace) => (
            <div
              key={workspace.id}
              className="flex items-center gap-3 border-b border-line px-4 py-3 transition-colors duration-150 last:border-b-0 hover:bg-surface"
            >
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <strong className="truncate text-sm font-medium text-fg">{workspace.name}</strong>
                <span className="font-mono text-[11px] tabular-nums text-fg-muted">
                  {workspace.columns.length} columns ·{" "}
                  {workspace.selectedRunIds.length} runs ·{" "}
                  {new Date(workspace.updatedAt).toLocaleDateString()}
                </span>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  loadWorkspace(workspace.id);
                  onClose();
                }}
              >
                Load
              </Button>
              <IconButton
                label={`Delete view ${workspace.name}`}
                size="sm"
                variant="secondary"
                className="hover:bg-negative-surface hover:text-negative"
                onClick={() => removeView(workspace)}
              >
                <Trash2 size={13} />
              </IconButton>
            </div>
          ))
        )}
      </div>

      <footer className="flex flex-none items-center justify-between gap-3 border-t border-line p-3">
        <span className="font-mono text-xs tabular-nums text-fg-muted">
          {workspaces.length} saved
        </span>
        <Button variant="primary" icon={<BookmarkPlus size={14} />} onClick={saveCurrentView}>
          Save current view
        </Button>
      </footer>
    </Modal>
  );
}
