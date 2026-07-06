import { backendFetch, backendRequest } from "@/app/backend";
import type { RunStatus, WorkflowPayload, WorkflowSchema } from "@/features/workflows/types";

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

type CheckpointResponse = Omit<Checkpoint, "metadata"> & {
  metadata?: Record<string, unknown>;
  metadata_?: Record<string, unknown>;
};

export async function fetchCheckpoints(): Promise<Checkpoint[]> {
  const rows = await backendRequest<CheckpointResponse[]>("/checkpoints");
  return rows.map(normalizeCheckpoint);
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

export function startCatalogDownload(item: { catalogKey: string; item: string; name: string }, schema?: WorkflowSchema): Promise<RunStatus> {
  const nodeSchema = schema?.nodes["CatalogDownload"];
  const payload: WorkflowPayload = {
    run_id: null,
    runner_id: null,
    nodes: [
      {
        id: "catalog_download",
        type: "CatalogDownload",
        x: 0,
        y: 0,
        params: {
          catalog_key: item.catalogKey,
          item: item.item,
        },
        runtime: nodeSchema ? structuredClone(nodeSchema.runtime_defaults) : {},
      },
    ],
    edges: [],
    context: {
      work_dir: "work",
      cache_dir: "cache",
      output_dir: "outputs",
      device: "cuda",
      config: {
        resources: {
          io: 2,
          cpu_workers: 2,
          accelerator: 1,
          vram_gb: 8,
        },
      },
      input_items: [],
    },
  };
  return backendRequest<RunStatus>("/graphs/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function normalizeCheckpoint(row: CheckpointResponse): Checkpoint {
  const { metadata_, metadata, ...rest } = row;
  return { ...rest, metadata: metadata ?? metadata_ ?? {} };
}
