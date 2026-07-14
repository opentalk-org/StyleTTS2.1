import { backendFetch, backendRequest } from "@/app/backend";
import type { RunStatus, WorkflowPayload } from "@/features/workflows/types";

export type Job = {
  run_id: string;
  name: string;
  state: "queued" | "running" | "stopping" | "stopped" | "succeeded" | "failed";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  updated_at: string;
  review_count: number;
};

export type ReviewState = "pending" | "approved" | "rejected";
export type ReviewTone = "neutral" | "success" | "warning" | "danger";

export type ReviewMetric = {
  key: string;
  label: string;
  value: string;
  numeric_value: number | null;
  tone: ReviewTone;
};

export type ReviewField = { key: string; label: string; value: string };

export type AudioSegmentReviewMedia = {
  kind: "audio_segment";
  audio_file_id: string;
  segment_id: string;
  start_seconds: number;
  end_seconds: number;
  duration_seconds: number;
  name: string;
};

export type ReviewItem = {
  key: string;
  title: string;
  subtitle: string | null;
  fields: ReviewField[];
  media: AudioSegmentReviewMedia[];
};

export type ReviewGroup = {
  key: string;
  title: string;
  explanation: string;
  tone: ReviewTone;
  items: ReviewItem[];
};

export type ReviewPayload = {
  metrics: ReviewMetric[];
  warnings: string[];
  groups: ReviewGroup[];
};

export type ReviewSummary = {
  id: string;
  producer_run_id: string;
  kind: string;
  title: string;
  state: ReviewState;
  continuation_run_id: string | null;
  created_at: string;
  decided_at: string | null;
};

export type WorkflowReview = ReviewSummary & {
  source_key: string;
  payload: ReviewPayload;
  continuation: { graph: unknown } | null;
  updated_at: string;
};

export type ReviewDecisionResponse = {
  review: WorkflowReview;
  continuation_run_id: string | null;
  continuation: RunStatus | null;
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

export function fetchReviews(runId: string): Promise<ReviewSummary[]> {
  return backendRequest<ReviewSummary[]>(`/reviews?run_id=${encodeURIComponent(runId)}`);
}

export function fetchReview(reviewId: string): Promise<WorkflowReview> {
  return backendRequest<WorkflowReview>(`/reviews/${encodeURIComponent(reviewId)}`);
}

export function decideReview(reviewId: string, decision: "approved" | "rejected"): Promise<ReviewDecisionResponse> {
  return backendRequest<ReviewDecisionResponse>(`/reviews/${encodeURIComponent(reviewId)}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision }),
  });
}

export async function deleteJob(runId: string): Promise<void> {
  const response = await backendFetch(`/jobs/${encodeURIComponent(runId)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export async function deleteJobs(runIds: string[]): Promise<void> {
  const response = await backendFetch(`/jobs/bulk-delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_ids: runIds }),
  });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export async function deleteAllJobs(): Promise<void> {
  const response = await backendFetch(`/jobs`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export async function renameJob(runId: string, name: string): Promise<void> {
  const response = await backendFetch(`/jobs/${encodeURIComponent(runId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export function stopJob(runId: string): Promise<RunStatus> {
  return backendRequest<RunStatus>(`/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
}
