import { backendRequest } from "@/app/backend";

export type Runner = {
  id: string;
  name: string;
  hostname: string;
  port: number;
  gpu_index: number | null;
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
};

export function fetchRunners(): Promise<RunnerPage> {
  return backendRequest<RunnerPage>("/runners");
}

export function createRunner(payload: RunnerRegisterPayload): Promise<Runner> {
  return backendRequest<Runner>("/runners", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
