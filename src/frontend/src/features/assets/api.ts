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

export function fetchFileAssets(type?: string): Promise<FileAsset[]> {
  const search = new URLSearchParams();
  if (type) search.set("type_", type);
  const suffix = search.size ? `?${search}` : "";
  return backendRequest<FileAsset[]>(`/assets/files${suffix}`);
}
