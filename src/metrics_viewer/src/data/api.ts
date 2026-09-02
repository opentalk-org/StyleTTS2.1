import { request } from "@/data/backend";
import type {
  ArrayMetricSeries,
  Artifact,
  ModelComponent,
  PlotQueryResult,
  Project,
  Run,
} from "@/shared/types";

export function listProjects(): Promise<Project[]> {
  return request<Project[]>("/api/projects");
}

export function listRuns(projectId: string): Promise<Run[]> {
  return request<Run[]>(`/api/projects/${encodeURIComponent(projectId)}/runs`);
}

export function getArtifacts(runIds: string[]): Promise<Artifact[]> {
  if (runIds.length === 0) return Promise.resolve([]);
  return request<Artifact[]>(`/api/artifacts?run_ids=${encodeURIComponent(runIds.join(","))}`);
}





export function runPlotsQuery(
  sql: string,
  projectId: string,
  selectedRunIds: string[],
): Promise<PlotQueryResult> {
  return request<PlotQueryResult>(`/api/projects/${encodeURIComponent(projectId)}/plots`, {
    method: "POST",
    body: JSON.stringify({ sql, runIds: selectedRunIds }),
  });
}

export function getModelGraph(runId: string): Promise<ModelComponent[]> {
  return request<ModelComponent[]>(`/api/runs/${encodeURIComponent(runId)}/model-graph`);
}

export function getArrayMetricNames(runId: string): Promise<string[]> {
  return request<string[]>(`/api/runs/${encodeURIComponent(runId)}/array-metrics/names`);
}

export function getArrayMetric(runId: string, name: string): Promise<ArrayMetricSeries> {
  return request<ArrayMetricSeries>(
    `/api/runs/${encodeURIComponent(runId)}/array-metrics?name=${encodeURIComponent(name)}`,
  );
}
