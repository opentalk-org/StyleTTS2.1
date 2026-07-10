import { backendRequest } from "@/app/backend";

export type StorageSettings = {
  id: string;
  bucket: string;
  folder: string;
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

export function testStorageSettings(payload: StorageSettingsPayload): Promise<{ ok: boolean }> {
  return backendRequest<{ ok: boolean }>("/settings/storage/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export type IntegrationSettings = {
  id: string;
  hf_token: string;
  openrouter_token: string;
  aim_url: string;
};

export type IntegrationSettingsPayload = Omit<IntegrationSettings, "id">;

export function fetchIntegrationSettings(): Promise<IntegrationSettings> {
  return backendRequest<IntegrationSettings>("/settings/integrations");
}

export function updateIntegrationSettings(payload: IntegrationSettingsPayload): Promise<IntegrationSettings> {
  return backendRequest<IntegrationSettings>("/settings/integrations", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
