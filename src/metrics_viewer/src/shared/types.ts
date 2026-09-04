export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type Scalar = string | number | boolean;
export type ArtifactKind = "audio" | "image" | "text" | "plot";
export type PanelTab = "charts" | "compare" | "media" | "graph";
export type XAxis = "step" | "relative" | "wall";

export interface Project {
  id: string;
  name: string;
  description: string;
  createdAt: number;
  lastRunAt: number;
  runCount: number;
  runningCount: number;
}

export interface Run {
  id: string;
  projectId: string;
  name: string;
  status: RunStatus;
  startedAt: number;
  endedAt: number;
  params: Record<string, Scalar>;
  summary: Record<string, number>;
}

export interface MetricPoint {
  runId: string;
  name: string;
  step: number;
  timestamp: number;
  value: number;
}

export interface Artifact {
  id: string;
  runId: string;
  name: string;
  step: number;
  timestamp: number;
  kind: ArtifactKind;
  contentType: string;
  sizeBytes: number;
  source: string;
}

export interface PlotQueryResult {
  plot: string[];
  runId: string[];
  x: number[];
  y: number[];
  /** Wall-clock ms and seconds since run start per point; null when the query has no such columns. */
  wall: number[] | null;
  rel: number[] | null;
  elapsedMs: number;
}

/** Settings shared by every chart; a chart's own PlotSettings can override them. */
export interface GlobalPlotSettings {
  xAxis: XAxis;
  /** EMA weight 0 (off) .. 0.99 applied to every chart whose smoothing is "inherit". */
  smoothing: number;
}

export interface PlotSettings {
  xAxis: "inherit" | XAxis;
  xScale: "linear" | "log";
  yScale: "linear" | "log";
  smoothing: "inherit" | "none" | "ema" | "mean";
  smoothingValue: number;
  renderMode: "line" | "scatter" | "line-scatter";
}

export interface ModelComponent {
  id: string;
  parent_id: string | null;
  name: string;
  module_type: string;
  parameter_names: string[];
  parameter_count: number;
}

export interface ArrayMetricSeries {
  name: string;
  steps: number[];
  timestamps: number[];
  values: number[][];
}

/** Compare tab: which runs and columns make up the table, and what the plot shows. */
export interface ComparePlotConfig {
  id: string;
  /** "run" or a column id; the x axis of the plot. */
  x: string;
  /** Column id plotted on y, or null while the plot is not configured yet. */
  y: string | null;
  logX: boolean;
  logY: boolean;
}

export interface CompareConfig {
  /** Column ids, or null for the automatic default. */
  columns: string[] | null;
  plots: ComparePlotConfig[];
}

export interface Workspace {
  id: string;
  name: string;
  projectId: string;
  selectedRunIds: string[];
  columns: string[];
  runColors: Record<string, string>;
  sql: string;
  globalPlot: GlobalPlotSettings;
  plotSettings: Record<string, PlotSettings>;
  hiddenPlots: string[];
  pinnedSections: string[];
  /** Chart names in user-chosen order; charts not listed keep the query order. */
  plotOrder: string[];
  compare: CompareConfig;
  tab: PanelTab;
  updatedAt: string;
}
