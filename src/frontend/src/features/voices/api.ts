import { backendFetch, backendRequest } from "@/app/backend";

export type Voice = {
  id: string;
  name: string;
  segments: number;
  datasets: string[];
};

export type VoiceQuery = {
  query: string;
  limit: number;
  offset: number;
};

export type VoicePage = {
  rows: Voice[];
  total: number;
};

type VoicePayload = {
  name: string;
};

export type VoiceDeleteRequest =
  | { mode: "ids"; ids: string[] }
  | { mode: "filter"; query: string };

export function fetchVoices(params: VoiceQuery): Promise<VoicePage> {
  const search = new URLSearchParams({
    query: params.query,
    limit: String(params.limit),
    offset: String(params.offset),
  });
  return backendRequest<VoicePage>(`/voices?${search}`);
}

export function createVoice(): Promise<Voice> {
  const payload: VoicePayload = { name: `speaker_${Date.now()}` };
  return backendRequest<Voice>("/voices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function renameVoice(id: string, name: string): Promise<Voice> {
  const payload: VoicePayload = { name };
  return backendRequest<Voice>(`/voices/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteVoice(id: string): Promise<void> {
  const response = await backendFetch(`/voices/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
}

export async function deleteVoices(ids: string[]): Promise<void> {
  await Promise.all(ids.map((id) => deleteVoice(id)));
}

export async function deleteMatchingVoices(query: string): Promise<void> {
  const search = new URLSearchParams({ query });
  const response = await backendFetch(`/voices?${search}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
}
