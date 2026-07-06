import { backendRequest } from "@/app/backend";

export type FileAsset = {
  id: string;
  name: string;
  path: string;
  size: number;
  content_hash: string;
  type_: string;
  metadata: Record<string, unknown>;
};

export type CreateTextFileAssetPayload = {
  name: string;
  type_: string;
  content: string;
  metadata?: Record<string, unknown>;
};

type FileAssetResponse = Omit<FileAsset, "metadata"> & {
  metadata?: Record<string, unknown>;
  metadata_?: Record<string, unknown>;
};

export async function fetchFileAssets(type?: string): Promise<FileAsset[]> {
  const search = new URLSearchParams();
  if (type) search.set("type_", type);
  const suffix = search.size ? `?${search}` : "";
  const rows = await backendRequest<FileAssetResponse[]>(`/assets/files${suffix}`);
  return rows.map(normalizeFileAsset);
}

export async function createTextFileAsset(payload: CreateTextFileAssetPayload): Promise<FileAsset> {
  const row = await backendRequest<FileAssetResponse>("/assets/files/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return normalizeFileAsset(row);
}

function normalizeFileAsset(row: FileAssetResponse): FileAsset {
  const { metadata_, metadata, ...rest } = row;
  return { ...rest, metadata: metadata ?? metadata_ ?? {} };
}
