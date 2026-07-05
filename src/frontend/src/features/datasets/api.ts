export type Dataset = {
  id: string;
  name: string;
  files: number;
};

type DatasetPayload = {
  name: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchDatasets(): Promise<Dataset[]> {
  return request<Dataset[]>("/datasets");
}

export function createDataset(name: string): Promise<Dataset> {
  const payload: DatasetPayload = { name };
  return request<Dataset>("/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteDataset(id: string): Promise<void> {
  const response = await fetch(`/datasets/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
}
