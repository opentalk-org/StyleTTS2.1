export interface ProjectRow {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  lastRunAt: string;
  runCount: string;
  runningCount: string;
}

export interface RunRow {
  id: string;
  projectId: string;
  name: string;
  status: string;
  startedAt: string;
  endedAt: string;
  trainingConfig: Record<string, unknown>;
}

export interface SummaryRow {
  runId: string;
  name: string;
  value: number;
}

export interface ArtifactRow {
  runId: string;
  step: string;
  timestamp: string;
  name: string;
  path: string;
  contentType: string;
  sizeBytes: string;
}

export interface PlotRow {
  plot: string;
  runId: string;
  x: number;
  y: number;
}

export type Scalar = string | number | boolean;

export function params(config: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(config).filter((entry): entry is [string, Scalar] =>
      typeof entry[1] === "string" || typeof entry[1] === "number" || typeof entry[1] === "boolean"
    ),
  );
}
