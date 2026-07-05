import { backendRequest } from "@/app/backend";
import type { WorkflowPayload } from "@/features/workflows/types";

export type Job = {
  run_id: string;
  name: string;
  state: "queued" | "running" | "stopping" | "stopped" | "succeeded" | "failed";
  graph_request: WorkflowPayload;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  updated_at: string;
};

export type JobPage = {
  rows: Job[];
  total: number;
};

export type JobQuery = {
  limit: number;
  offset: number;
};

export function fetchJobs(params: JobQuery): Promise<JobPage> {
  const search = new URLSearchParams({ limit: String(params.limit), offset: String(params.offset) });
  return backendRequest<JobPage>(`/jobs?${search}`);
}

export function fetchJobGraph(runId: string): Promise<WorkflowPayload> {
  return backendRequest<WorkflowPayload>(`/jobs/${encodeURIComponent(runId)}/graph`);
}
