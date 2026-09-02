export type RunStatus = "awaiting" | "running" | "succeeded" | "failed" | "cancelled";
export type Scalar = string | number | boolean;
export type ArtifactKind = "audio" | "image" | "text" | "plot";

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
  elapsedMs: number;
}


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

  sql: string;
  plotSettings: Record<string, PlotSettings>;
  hiddenPlots: string[];
  updatedAt: string;
}
