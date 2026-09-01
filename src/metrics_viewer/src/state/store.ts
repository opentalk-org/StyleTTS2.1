import { create } from "zustand";

import type { PlotSettings, Workspace } from "@/shared/types";

const STORAGE_KEY = "runflow.metrics.workspaces.v4";
const DEFAULT_COLUMNS = [
  "name",
  "status",
  "startedAt",
  "duration",
  "param:decoder",
  "metric:val/mel_loss",
];

/** Applied to any plot the query produces that has no saved settings yet. */
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

/** The query that decides which plots exist. Every view starts from this. */
export const DEFAULT_SQL = `SELECT
  name      AS plot,
  run_id,
  step      AS x,
  value     AS y
FROM metrics(
  names => ['val/mel_loss', 'train/generator_total', 'system/gpu_utilization_percent'],
  runs  => selected()
)`;

interface ViewerState {
  projectId: string | null;
  selectedRunIds: string[];
  columns: string[];
  /** Per-run plot color overrides, keyed by run id. Unset runs fall back to the palette. */
  runColors: Record<string, string>;
  /** Display settings per plot name, as produced by the query. */
  plotSettings: Record<string, PlotSettings>;
  /** Plots dropped from this view without editing the query. */
  hiddenPlots: string[];
  /** The editor buffer. Editing it does not move the plots. */
  sql: string;
  /** The query the plots are actually drawn from; only Run promotes sql into it. */
  runningSql: string;
  workspaces: Workspace[];
  selectProject: (id: string | null) => void;
  toggleRun: (id: string) => void;
  selectRuns: (ids: string[]) => void;
  setColumns: (columns: string[]) => void;
  setRunColor: (runId: string, color: string | null) => void;
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
  columns: DEFAULT_COLUMNS,
  runColors: {},
  plotSettings: {},
  hiddenPlots: [],
  sql: DEFAULT_SQL,
  runningSql: DEFAULT_SQL,
  workspaces: loadWorkspaces(),
  selectProject: (projectId) => set({ projectId, selectedRunIds: [] }),
  toggleRun: (id) =>
    set((state) => ({
      selectedRunIds: state.selectedRunIds.includes(id)
        ? state.selectedRunIds.filter((item) => item !== id)
        : [...state.selectedRunIds, id],
    })),
  selectRuns: (selectedRunIds) => set({ selectedRunIds }),
  setColumns: (columns) => set({ columns }),
  setRunColor: (runId, color) =>
    set((state) => {
      const runColors = { ...state.runColors };
      if (color === null) delete runColors[runId];
      else runColors[runId] = color;
      return { runColors };
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
