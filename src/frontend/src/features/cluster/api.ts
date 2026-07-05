export type Runner = {
  id: string;
  name: string;
  hostname: string;
  port: number;
  gpu_index: number | null;
  resources: Record<string, number>;
  online: boolean;
  stale: boolean;
  busy: boolean;
  active_run_ids: string[];
  process_id: number | null;
  last_seen_at: string | null;
};

type RunnerPage = {
  rows: Runner[];
  total: number;
};

export type RunnerRegisterPayload = {
  name: string;
  hostname: string;
  port: number;
  gpu_index: number | null;
  resources: Record<string, number>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
  return response.json() as Promise<T>;
}

export function fetchRunners(): Promise<RunnerPage> {
  return request<RunnerPage>("/runners");
}

export function createRunner(payload: RunnerRegisterPayload): Promise<Runner> {
  return request<Runner>("/runners", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
