import { backendFetch, backendRequest } from "@/app/backend";

export type Checkpoint = {
  id: string;
  name: string;
  path: string;
  size: number;
  content_hash: string;
  type_: string;
  metadata: Record<string, unknown>;
  job_id: string | null;
};

export function fetchCheckpoints(): Promise<Checkpoint[]> {
  return backendRequest<Checkpoint[]>("/checkpoints");
}

export function renameCheckpoint(checkpoint: Checkpoint, name: string): Promise<Checkpoint> {
  return backendRequest<Checkpoint>(`/checkpoints/${checkpoint.id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      folder_path: null,
      type_: checkpoint.type_,
      metadata: checkpoint.metadata,
      job_id: checkpoint.job_id,
    }),
  });
}

export async function deleteCheckpoint(id: string): Promise<void> {
  const response = await backendFetch(`/checkpoints/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}
