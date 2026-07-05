export type StorageSettings = {
  id: string;
  bucket: string;
  endpoint_url: string;
  region_name: string;
  access_key_id: string;
  secret_access_key: string;
};

export type StorageSettingsPayload = Omit<StorageSettings, "id">;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
  return response.json() as Promise<T>;
}

export function fetchStorageSettings(): Promise<StorageSettings> {
  return request<StorageSettings>("/settings/storage");
}

export function updateStorageSettings(payload: StorageSettingsPayload): Promise<StorageSettings> {
  return request<StorageSettings>("/settings/storage", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
