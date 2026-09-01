export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type Scalar = string | number | boolean;
export type ArtifactKind = "audio" | "image" | "text" | "plot";

export interface Project {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  lastRunAt: string;
  runCount: number;
  runningCount: number;
}

export interface Run {
  id: string;
  projectId: string;
  name: string;
  status: RunStatus;
  startedAt: string;
  endedAt: string | null;
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

/** One point of one series, as returned by the plots query. */
export interface PlotRow {
  /** Chart this point belongs to; one chart per distinct value. */
  plot: string;
  runId: string;
  x: number;
  y: number;
}

export interface PlotQueryResult {
  rows: PlotRow[];
  elapsedMs: number;
  readRows: number;
  /** What the x column was selected from, so axes can be labelled honestly. */
  xLabel: string;
}

/** Display settings for one chart, kept per plot name rather than per chart id. */
export interface PlotSettings {
  xScale: "linear" | "log";
  yScale: "linear" | "log";
  smoothing: "none" | "ema" | "mean";
  smoothingValue: number;
  renderMode: "line" | "scatter" | "line-scatter";
  rawOpacity: number;
  smoothOpacity: number;
  showLegend: boolean;
}

export interface Workspace {
  id: string;
  name: string;
  projectId: string;
  selectedRunIds: string[];
  columns: string[];
  runColors: Record<string, string>;
  /** The query that defines which plots exist. */
  sql: string;
  plotSettings: Record<string, PlotSettings>;
  hiddenPlots: string[];
  updatedAt: string;
}
