import { backendFetch, backendRequest, backendResourceUrl } from "@/app/backend";

export type Artifact = {
  id: string;
  name: string;
  path: string;
  size: number;
  content_hash: string;
  type_: string;
  metadata: Record<string, unknown>;
};

type ArtifactResponse = Omit<Artifact, "metadata"> & {
  metadata?: Record<string, unknown>;
  metadata_?: Record<string, unknown>;
};

export async function fetchArtifacts(): Promise<Artifact[]> {
  const rows = await backendRequest<ArtifactResponse[]>("/artifacts");
  return rows.map(normalizeArtifact);
}

export function artifactContentUrl(id: string): string {
  return backendResourceUrl(`/artifacts/${encodeURIComponent(id)}/content`);
}

export async function deleteArtifact(id: string): Promise<void> {
  const response = await backendFetch(`/artifacts/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

function normalizeArtifact(row: ArtifactResponse): Artifact {
  const { metadata_, metadata, ...rest } = row;
  return { ...rest, metadata: metadata ?? metadata_ ?? {} };
}
