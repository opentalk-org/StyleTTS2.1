import type { NodeRunSnapshot, RunSnapshot, RunStatus, RunnerStatus, WorkflowPayload, WorkflowSchema } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
  return response.json() as Promise<T>;
}

export function fetchWorkflowSchema(): Promise<WorkflowSchema> {
  return request<WorkflowSchema>("/schema");
}

export function startGraph(payload: WorkflowPayload): Promise<RunStatus> {
  return request<RunStatus>("/graphs/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchRuns(): Promise<RunnerStatus> {
  return request<RunnerStatus>("/runs");
}

export function fetchRun(runId: string): Promise<RunStatus> {
  return request<RunStatus>(`/runs/${encodeURIComponent(runId)}`);
}

export function fetchRunSnapshot(runId: string): Promise<RunSnapshot> {
  return request<RunSnapshot>(`/runs/${encodeURIComponent(runId)}/snapshot`);
}

export function fetchRunGraph(runId: string): Promise<WorkflowPayload> {
  return request<WorkflowPayload>(`/runs/${encodeURIComponent(runId)}/graph`);
}

export function stopRun(runId: string): Promise<RunStatus> {
  return request<RunStatus>(`/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
}

export function loadRunNode(runId: string, nodeId: string): Promise<RunStatus> {
  return request<RunStatus>(`/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/load`, { method: "POST" });
}

export function unloadRunNode(runId: string, nodeId: string): Promise<RunStatus> {
  return request<RunStatus>(`/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/unload`, { method: "POST" });
}

export function fetchNodeLog(runId: string, nodeId: string): Promise<{ content: string; truncated: boolean; error: string | null }> {
  return request(`/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/logs`);
}

export function nodeSnapshot(snapshot: RunSnapshot | undefined, nodeId: string): NodeRunSnapshot | undefined {
  return snapshot?.nodes.find((node) => node.node_id === nodeId);
}
