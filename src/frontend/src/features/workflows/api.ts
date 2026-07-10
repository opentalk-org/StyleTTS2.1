import { backendFetch, backendRequest } from "@/app/backend";

import type { NodeRunSnapshot, RunErrorEvent, RunSnapshot, RunStatus, RunnerStatus, SaveWorkflowPayload, SavedWorkflow, WorkflowPayload, WorkflowSchema } from "./types";

export function fetchWorkflowSchema(): Promise<WorkflowSchema> {
  return backendRequest<WorkflowSchema>("/schema");
}

export function fetchSavedWorkflows(): Promise<SavedWorkflow[]> {
  return backendRequest<SavedWorkflow[]>("/workflows");
}

export function fetchExampleWorkflows(): Promise<SavedWorkflow[]> {
  return backendRequest<SavedWorkflow[]>("/workflows/examples");
}

export function saveWorkflow(payload: SaveWorkflowPayload): Promise<SavedWorkflow> {
  return backendRequest<SavedWorkflow>("/workflows", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteWorkflow(id: string): Promise<void> {
  const response = await backendFetch(`/workflows/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export function startGraph(payload: WorkflowPayload): Promise<RunStatus> {
  return backendRequest<RunStatus>("/graphs/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchRuns(): Promise<RunnerStatus> {
  return backendRequest<RunnerStatus>("/runs");
}

export function fetchRun(runId: string): Promise<RunStatus> {
  return backendRequest<RunStatus>(`/runs/${encodeURIComponent(runId)}`);
}

export function fetchRunSnapshot(runId: string): Promise<RunSnapshot> {
  return backendRequest<RunSnapshot>(`/runs/${encodeURIComponent(runId)}/snapshot`);
}

export function fetchRunErrors(runId: string): Promise<RunErrorEvent[]> {
  return backendRequest<RunErrorEvent[]>(`/runs/${encodeURIComponent(runId)}/errors`);
}

export function fetchRunGraph(runId: string): Promise<WorkflowPayload> {
  return backendRequest<WorkflowPayload>(`/runs/${encodeURIComponent(runId)}/graph`);
}

export function stopRun(runId: string): Promise<RunStatus> {
  return backendRequest<RunStatus>(`/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
}

export function loadRunNode(runId: string, nodeId: string): Promise<RunStatus> {
  return backendRequest<RunStatus>(`/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/load`, { method: "POST" });
}

export function unloadRunNode(runId: string, nodeId: string): Promise<RunStatus> {
  return backendRequest<RunStatus>(`/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/unload`, { method: "POST" });
}

export function fetchNodeLog(runId: string, nodeId: string): Promise<{ content: string; truncated: boolean; error: string | null }> {
  return backendRequest(`/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/logs`);
}

export function nodeSnapshot(snapshot: RunSnapshot | undefined, nodeId: string): NodeRunSnapshot | undefined {
  return snapshot?.nodes.find((node) => node.node_id === nodeId);
}
