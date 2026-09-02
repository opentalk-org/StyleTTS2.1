import { request } from "@/data/backend";
import type { Artifact, PlotQueryResult, Project, Run } from "@/shared/types";

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
