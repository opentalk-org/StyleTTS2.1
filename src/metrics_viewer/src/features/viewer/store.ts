import { create } from "zustand";

import { defaultRunColumns } from "@/shared/metrics";
import type { PlotSettings, Run, Workspace } from "@/shared/types";

const STORAGE_KEY = "runflow.metrics.workspaces.v4";
const STARS_KEY = "runflow.metrics.stars.v1";

export const DEFAULT_PLOT_SETTINGS: PlotSettings = {
  xScale: "linear",
  yScale: "linear",
  smoothing: "none",
  smoothingValue: 0.75,
  renderMode: "line",
  rawOpacity: 0.22,
  smoothOpacity: 0.95,
  showLegend: true,
};

export const DEFAULT_SQL = `SELECT
  plot,
  run_id,
  point.1 AS x,
  point.2 AS y
FROM (
  SELECT
    name AS plot,
    run_id,
    arrayJoin(largestTriangleThreeBuckets(1000)(step, value)) AS point
  FROM metrics
  WHERE run_id IN {run_ids:Array(UUID)}
  GROUP BY plot, run_id
)`;

interface ViewerState {
  projectId: string | null;
  selectedRunIds: string[];
  columns: string[];
  columnsInitialized: boolean;
  runColors: Record<string, string>;
  plotSettings: Record<string, PlotSettings>;
  hiddenPlots: string[];
  starredRunIds: string[];
  sql: string;
  runningSql: string;
  workspaces: Workspace[];
  selectProject: (id: string | null) => void;
  toggleRun: (id: string) => void;
  selectRuns: (ids: string[]) => void;
  initializeColumns: (runs: Run[]) => void;
  setColumns: (columns: string[]) => void;
  setRunColor: (runId: string, color: string | null) => void;
  toggleStar: (runId: string) => void;
  setSql: (sql: string) => void;
  commitSql: () => void;
  updatePlot: (plot: string, patch: Partial<PlotSettings>) => void;
  resetPlot: (plot: string) => void;
  togglePlotHidden: (plot: string) => void;
  showAllPlots: () => void;
  saveWorkspace: (name: string) => void;
  loadWorkspace: (id: string) => void;
  deleteWorkspace: (id: string) => void;
}

function loadStars(): string[] {
  try {
    return JSON.parse(localStorage.getItem(STARS_KEY) ?? "[]") as string[];
  } catch {
    localStorage.removeItem(STARS_KEY);
    return [];
  }
}

function loadWorkspaces(): Workspace[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as Workspace[];
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return [];
  }
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  projectId: null,
  selectedRunIds: [],
  columns: defaultRunColumns([]),
  columnsInitialized: false,
  runColors: {},
  plotSettings: {},
  hiddenPlots: [],
  starredRunIds: loadStars(),
  sql: DEFAULT_SQL,
  runningSql: DEFAULT_SQL,
  workspaces: loadWorkspaces(),
  selectProject: (projectId) =>
    set({
      projectId,
      selectedRunIds: [],
      columns: defaultRunColumns([]),
      columnsInitialized: false,
    }),
  toggleRun: (id) =>
    set((state) => ({
      selectedRunIds: state.selectedRunIds.includes(id)
        ? state.selectedRunIds.filter((item) => item !== id)
        : [...state.selectedRunIds, id],
    })),
  selectRuns: (selectedRunIds) => set({ selectedRunIds }),
  initializeColumns: (runs) =>
    set((state) =>
      state.columnsInitialized || runs.length === 0
        ? state
        : { columns: defaultRunColumns(runs), columnsInitialized: true },
    ),
  setColumns: (columns) => set({ columns, columnsInitialized: true }),
  setRunColor: (runId, color) =>
    set((state) => {
      const runColors = { ...state.runColors };
      if (color === null) delete runColors[runId];
      else runColors[runId] = color;
      return { runColors };
    }),
  toggleStar: (runId) =>
    set((state) => {
      const starredRunIds = state.starredRunIds.includes(runId)
        ? state.starredRunIds.filter((item) => item !== runId)
        : [...state.starredRunIds, runId];
      localStorage.setItem(STARS_KEY, JSON.stringify(starredRunIds));
      return { starredRunIds };
    }),
  setSql: (sql) => set({ sql }),
  commitSql: () => set((state) => ({ runningSql: state.sql })),
  updatePlot: (plot, patch) =>
    set((state) => ({
      plotSettings: {
        ...state.plotSettings,
        [plot]: { ...DEFAULT_PLOT_SETTINGS, ...state.plotSettings[plot], ...patch },
      },
    })),
  resetPlot: (plot) =>
    set((state) => {
      const plotSettings = { ...state.plotSettings };
      delete plotSettings[plot];
      return { plotSettings };
    }),
  togglePlotHidden: (plot) =>
    set((state) => ({
      hiddenPlots: state.hiddenPlots.includes(plot)
        ? state.hiddenPlots.filter((item) => item !== plot)
        : [...state.hiddenPlots, plot],
    })),
  showAllPlots: () => set({ hiddenPlots: [] }),
  saveWorkspace: (name) => {
    const state = get();
    if (state.projectId === null) return;
    const workspace: Workspace = {
      id: crypto.randomUUID(),
      name,
      projectId: state.projectId,
      selectedRunIds: state.selectedRunIds,
      columns: state.columns,
      runColors: state.runColors,
      sql: state.sql,
      plotSettings: state.plotSettings,
      hiddenPlots: state.hiddenPlots,
      updatedAt: new Date().toISOString(),
    };
    const workspaces = [workspace, ...state.workspaces];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(workspaces));
    set({ workspaces });
  },
  loadWorkspace: (id) => {
    const workspace = get().workspaces.find((item) => item.id === id);
    if (workspace === undefined) return;
    set({
      projectId: workspace.projectId,
      selectedRunIds: workspace.selectedRunIds,
      columns: workspace.columns,
      columnsInitialized: true,
      runColors: workspace.runColors ?? {},
      sql: workspace.sql,
      runningSql: workspace.sql,
      plotSettings: workspace.plotSettings ?? {},
      hiddenPlots: workspace.hiddenPlots ?? [],
    });
  },
  deleteWorkspace: (id) => {
    const workspaces = get().workspaces.filter((item) => item.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(workspaces));
    set({ workspaces });
  },
}));
