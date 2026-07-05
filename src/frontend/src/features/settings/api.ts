import { backendRequest } from "@/app/backend";

export type StorageSettings = {
  id: string;
  bucket: string;
  endpoint_url: string;
  region_name: string;
  access_key_id: string;
  secret_access_key: string;
};

export type StorageSettingsPayload = Omit<StorageSettings, "id">;

export function fetchStorageSettings(): Promise<StorageSettings> {
  return backendRequest<StorageSettings>("/settings/storage");
}

export function updateStorageSettings(payload: StorageSettingsPayload): Promise<StorageSettings> {
  return backendRequest<StorageSettings>("/settings/storage", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
