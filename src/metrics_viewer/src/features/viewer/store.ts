import { create } from "zustand";

import { defaultRunColumns, defaultRunColumnsFromMetrics } from "@/shared/metrics";
import type { CompareConfig, ComparePlotConfig, ComparePlotKind, GlobalPlotSettings, PanelColumns, PanelTab, PlotSettings, Run, RunScope, Workspace } from "@/shared/types";

const STORAGE_KEY = "runflow.metrics.workspaces.v5";
const STARS_KEY = "runflow.metrics.stars.v1";
const THEME_KEY = "runflow.metrics.theme.v1";

export type Theme = "dark" | "light";

export const DEFAULT_PLOT_SETTINGS: PlotSettings = {
  xAxis: "inherit",
  xScale: "linear",
  yScale: "linear",
  smoothing: "inherit",
  smoothingValue: 0.75,
  renderMode: "line",
};

export const DEFAULT_GLOBAL_PLOT: GlobalPlotSettings = { xAxis: "step", smoothing: 0 };

export const DEFAULT_COMPARE: CompareConfig = { columns: null, plots: [] };

/** LTTB keeps each run/metric series bounded without materializing raw points in the client. */
export const DEFAULT_SQL = `SELECT
  sampled.name AS plot,
  sampled.run_id AS run_id,
  point.1 AS x,
  point.2 AS y,
  toUnixTimestamp64Milli(metric_timestamp) AS wall,
  dateDiff('millisecond', started_at, metric_timestamp) / 1000.0 AS rel
FROM (
  SELECT
    name,
    run_id,
    largestTriangleThreeBuckets(1000)(step, value) AS points
  FROM (
    SELECT run_id, name, step, argMax(value, timestamp) AS value, max(timestamp) AS timestamp
    FROM metrics
    WHERE run_id IN {run_ids:Array(UUID)}
    GROUP BY run_id, name, step
  )
  GROUP BY run_id, name
) AS sampled
ARRAY JOIN points AS point
INNER JOIN (
  SELECT run_id, name, step, max(timestamp) AS metric_timestamp
  FROM metrics
  WHERE run_id IN {run_ids:Array(UUID)}
  GROUP BY run_id, name, step
) AS times ON times.run_id = sampled.run_id AND times.name = sampled.name AND times.step = toUInt64(point.1)
INNER JOIN (
  SELECT run_id, min(timestamp) AS started_at
  FROM run_status
  WHERE run_id IN {run_ids:Array(UUID)}
  GROUP BY run_id
) AS starts ON starts.run_id = sampled.run_id`;

export function isDefaultSql(sql: string): boolean {
  return sql === DEFAULT_SQL;
}

export interface HydrateInput {
  projectId: string | null;
  selectedRunIds: string[];
  tab: PanelTab;
  view: Pick<ViewerState, "runScope" | "columns" | "runColorOverrides" | "globalPlot" | "plotSettings" | "hiddenPlots" | "pinnedSections" | "plotOrder" | "compare" | "sql" | "chartColumns" | "mediaColumns">;
}

interface ViewerState {
  theme: Theme;
  projectId: string | null;
  selectedRunIds: string[];
  /** Charts and compare draw the selected runs, or those plus the runs they resumed from. */
  runScope: RunScope;
  columns: string[];
  columnsInitialized: boolean;
  runColorOverrides: Record<string, string>;
  starredRunIds: string[];
  tab: PanelTab;
  globalPlot: GlobalPlotSettings;
  plotSettings: Record<string, PlotSettings>;
  hiddenPlots: string[];
  pinnedSections: string[];
  plotOrder: string[];
  compare: CompareConfig;
  sql: string;
  runningSql: string;
  chartColumns: PanelColumns;
  mediaColumns: PanelColumns;
  workspaces: Workspace[];
  setTheme: (theme: Theme) => void;
  hydrate: (input: HydrateInput) => void;
  selectProject: (id: string | null) => void;
  setTab: (tab: PanelTab) => void;
  setRunScope: (runScope: RunScope) => void;
  toggleRun: (id: string) => void;
  focusRun: (id: string, additive: boolean) => void;
  selectRuns: (ids: string[]) => void;
  initializeColumns: (metrics: string[]) => void;
  setColumns: (columns: string[]) => void;
  setChartColumns: (columns: PanelColumns) => void;
  setMediaColumns: (columns: PanelColumns) => void;
  setRunColor: (runId: string, color: string | null) => void;
  toggleStar: (runId: string) => void;
  setGlobalPlot: (patch: Partial<GlobalPlotSettings>) => void;
  setSql: (sql: string) => void;
  commitSql: () => void;
  resetSql: () => void;
  updatePlot: (plot: string, patch: Partial<PlotSettings>) => void;
  resetPlot: (plot: string) => void;
  resetAllPlots: () => void;
  togglePlotHidden: (plot: string) => void;
  showAllPlots: () => void;
  togglePinnedSection: (section: string) => void;
  /** Records the order of one group of charts; other charts keep their existing order. */
  setPlotOrder: (names: string[]) => void;
  setCompareColumns: (columns: string[] | null) => void;
  addComparePlot: (kind: ComparePlotKind) => void;
  updateComparePlot: (id: string, patch: Partial<ComparePlotConfig>) => void;
  removeComparePlot: (id: string) => void;
  saveWorkspace: (name: string) => Workspace;
  renameWorkspace: (id: string, name: string) => void;
  loadWorkspace: (id: string) => void;
  deleteWorkspace: (id: string) => void;
}

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    localStorage.removeItem(key);
    return fallback;
  }
}

function loadTheme(): Theme {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function toggleItem(items: string[], item: string): string[] {
  return items.includes(item) ? items.filter((candidate) => candidate !== item) : [...items, item];
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  theme: loadTheme(),
  projectId: null,
  selectedRunIds: [],
  runScope: "selected",
  columns: defaultRunColumns([]),
  columnsInitialized: false,
  runColorOverrides: {},
  starredRunIds: loadJson<string[]>(STARS_KEY, []),
  tab: "charts",
  globalPlot: DEFAULT_GLOBAL_PLOT,
  plotSettings: {},
  hiddenPlots: [],
  pinnedSections: [],
  plotOrder: [],
  compare: DEFAULT_COMPARE,
  sql: DEFAULT_SQL,
  runningSql: DEFAULT_SQL,
  chartColumns: "auto",
  mediaColumns: "auto",
  workspaces: loadJson<Workspace[]>(STORAGE_KEY, []),
  setTheme: (theme) => {
    localStorage.setItem(THEME_KEY, theme);
    set({ theme });
  },
  hydrate: ({ projectId, selectedRunIds, tab, view }) =>
    set({
      projectId,
      selectedRunIds,
      tab,
      ...view,
      columnsInitialized: true,
      runningSql: view.sql,
    }),
  selectProject: (projectId) =>
    set({ projectId, selectedRunIds: [], columns: defaultRunColumns([]), columnsInitialized: false, compare: DEFAULT_COMPARE }),
  setTab: (tab) => set({ tab }),
  setRunScope: (runScope) => set({ runScope }),
  toggleRun: (id) => set((state) => ({ selectedRunIds: toggleItem(state.selectedRunIds, id) })),
  focusRun: (id, additive) =>
    set((state) => ({
      selectedRunIds: additive ? [...new Set([...state.selectedRunIds, id])] : [id],
    })),
  selectRuns: (selectedRunIds) => set({ selectedRunIds }),
  initializeColumns: (metrics) =>
    set((state) =>
      state.columnsInitialized
        ? state
        : { columns: defaultRunColumnsFromMetrics(metrics), columnsInitialized: true },
    ),
  setColumns: (columns) => set({ columns, columnsInitialized: true }),
  setChartColumns: (chartColumns) => set({ chartColumns }),
  setMediaColumns: (mediaColumns) => set({ mediaColumns }),
  setRunColor: (runId, color) =>
    set((state) => {
      const runColorOverrides = { ...state.runColorOverrides };
      if (color === null) delete runColorOverrides[runId];
      else runColorOverrides[runId] = color;
      return { runColorOverrides };
    }),
  toggleStar: (runId) =>
    set((state) => {
      const starredRunIds = toggleItem(state.starredRunIds, runId);
      localStorage.setItem(STARS_KEY, JSON.stringify(starredRunIds));
      return { starredRunIds };
    }),
  setGlobalPlot: (patch) => set((state) => ({ globalPlot: { ...state.globalPlot, ...patch } })),
  setSql: (sql) => set({ sql }),
  commitSql: () => set((state) => ({ runningSql: state.sql })),
  resetSql: () => set({ sql: DEFAULT_SQL, runningSql: DEFAULT_SQL }),
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
  resetAllPlots: () => set({ plotSettings: {}, globalPlot: DEFAULT_GLOBAL_PLOT, plotOrder: [] }),
  togglePlotHidden: (plot) => set((state) => ({ hiddenPlots: toggleItem(state.hiddenPlots, plot) })),
  showAllPlots: () => set({ hiddenPlots: [] }),
  togglePinnedSection: (section) =>
    set((state) => ({ pinnedSections: toggleItem(state.pinnedSections, section) })),
  setPlotOrder: (names) =>
    set((state) => ({ plotOrder: [...state.plotOrder.filter((name) => !names.includes(name)), ...names] })),
  setCompareColumns: (columns) => set((state) => ({ compare: { ...state.compare, columns } })),
  addComparePlot: (kind) =>
    set((state) => ({
      compare: {
        ...state.compare,
        plots: [...state.compare.plots, { id: crypto.randomUUID(), kind, x: "run", y: null, logX: false, logY: false, dims: null, colorBy: null }],
      },
    })),
  updateComparePlot: (id, patch) =>
    set((state) => ({
      compare: {
        ...state.compare,
        plots: state.compare.plots.map((plot) => (plot.id === id ? { ...plot, ...patch } : plot)),
      },
    })),
  removeComparePlot: (id) =>
    set((state) => ({ compare: { ...state.compare, plots: state.compare.plots.filter((plot) => plot.id !== id) } })),
  saveWorkspace: (name) => {
    const state = get();
    const workspace: Workspace = {
      id: crypto.randomUUID(),
      name,
      projectId: state.projectId as string,
      selectedRunIds: state.selectedRunIds,
      columns: state.columns,
      runColors: state.runColorOverrides,
      sql: state.sql,
      globalPlot: state.globalPlot,
      plotSettings: state.plotSettings,
      hiddenPlots: state.hiddenPlots,
      pinnedSections: state.pinnedSections,
      plotOrder: state.plotOrder,
      compare: state.compare,
      tab: state.tab,
      runScope: state.runScope,
      updatedAt: new Date().toISOString(),
    };
    const workspaces = [workspace, ...state.workspaces];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(workspaces));
    set({ workspaces });
    return workspace;
  },
  renameWorkspace: (id, name) => {
    const workspaces = get().workspaces.map((item) => (item.id === id ? { ...item, name } : item));
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
      runColorOverrides: workspace.runColors,
      sql: workspace.sql,
      runningSql: workspace.sql,
      globalPlot: workspace.globalPlot,
      plotSettings: workspace.plotSettings,
      hiddenPlots: workspace.hiddenPlots,
      pinnedSections: workspace.pinnedSections,
      plotOrder: workspace.plotOrder,
      compare: workspace.compare,
      tab: workspace.tab,
      runScope: workspace.runScope ?? "selected",
    });
  },
  deleteWorkspace: (id) => {
    const workspaces = get().workspaces.filter((item) => item.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(workspaces));
    set({ workspaces });
  },
}));
