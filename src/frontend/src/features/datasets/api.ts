import { backendFetch, backendRequest } from "@/app/backend";

export type Dataset = {
  id: string;
  name: string;
  files: number;
};

type DatasetPayload = {
  name: string;
};

export function fetchDatasets(): Promise<Dataset[]> {
  return backendRequest<Dataset[]>("/datasets");
}

export function createDataset(name: string): Promise<Dataset> {
  const payload: DatasetPayload = { name };
  return backendRequest<Dataset>("/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteDataset(id: string): Promise<void> {
  const response = await backendFetch(`/datasets/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
}
