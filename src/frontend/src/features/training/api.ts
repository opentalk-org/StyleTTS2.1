import { backendRequest } from "@/app/backend";

export type TrainingConfig = {
  id: string;
  name: string;
  type_: string;
  metadata: Record<string, unknown>;
};

type TrainingConfigResponse = Omit<TrainingConfig, "metadata"> & {
  metadata?: Record<string, unknown>;
  metadata_?: Record<string, unknown>;
};

export type CreateTrainingConfigPayload = {
  name: string;
  type_: string;
  metadata: Record<string, unknown>;
};

export async function fetchTrainingConfigs(type: string): Promise<TrainingConfig[]> {
  const search = new URLSearchParams({ type_: type });
  const rows = await backendRequest<TrainingConfigResponse[]>(`/assets/configs?${search}`);
  return rows.map(normalizeTrainingConfig);
}

export async function createTrainingConfig(payload: CreateTrainingConfigPayload): Promise<TrainingConfig> {
  const row = await backendRequest<TrainingConfigResponse>("/assets/configs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return normalizeTrainingConfig(row);
}

function normalizeTrainingConfig(row: TrainingConfigResponse): TrainingConfig {
  const { metadata_, metadata, ...rest } = row;
  return { ...rest, metadata: metadata ?? metadata_ ?? {} };
}
