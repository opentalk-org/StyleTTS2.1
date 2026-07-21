import { backendFetch, backendRequest } from "@/app/backend";

export type Speaker = {
  id: string;
  audio_files: number;
  segments: number;
  datasets: string[];
};

export type SpeakerQuery = {
  query: string;
  limit: number;
  offset: number;
};

export type SpeakerPage = {
  rows: Speaker[];
  total: number;
};

type SpeakerRename = {
  speaker_id: string;
};

export type SpeakerDeleteRequest =
  | { mode: "ids"; ids: string[] }
  | { mode: "filter"; query: string };

export function fetchSpeakers(params: SpeakerQuery): Promise<SpeakerPage> {
  const search = new URLSearchParams({
    query: params.query,
    limit: String(params.limit),
    offset: String(params.offset),
  });
  return backendRequest<SpeakerPage>(`/speakers?${search}`);
}

export async function renameSpeaker(id: string, speakerId: string): Promise<void> {
  const payload: SpeakerRename = { speaker_id: speakerId };
  const response = await backendFetch(`/speakers/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export async function deleteSpeaker(id: string): Promise<void> {
  const response = await backendFetch(`/speakers/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
}

export async function deleteSpeakers(ids: string[]): Promise<void> {
  await Promise.all(ids.map((id) => deleteSpeaker(id)));
}

export async function deleteMatchingSpeakers(query: string): Promise<void> {
  const search = new URLSearchParams({ query });
  const response = await backendFetch(`/speakers?${search}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
}
