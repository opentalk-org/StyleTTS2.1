import { backendFetch, backendRequest } from "@/app/backend";
import type { AudioAnnotations } from "@/features/audio/api";

export type MosAudio = {
  id: string;
  name: string;
  duration: number;
  annotations: AudioAnnotations;
};

export type MosPair = {
  dataset_id: string;
  audio_a: MosAudio;
  audio_b: MosAudio;
};

export type MosRatingRequest = {
  dataset_id: string;
  audio_a_id: string;
  audio_b_id: string;
  preferred_audio_id: string;
  score_a: number;
  score_b: number;
};

export type MosRating = {
  id: string;
  audio_a_id: string;
  audio_b_id: string;
  preferred_audio_id: string;
  score_a: number;
  score_b: number;
  created_at: string;
};

export type MosRatingUpdateRequest = {
  preferred_audio_id: string;
  score_a: number;
  score_b: number;
};

export type MosHistoryItem = MosRating & {
  audio_a: MosAudio;
  audio_b: MosAudio;
  can_modify: boolean;
};

export type MosRatingPage = {
  rows: MosHistoryItem[];
  total: number;
  limit: number;
  offset: number;
};

export function fetchMosPair(datasetIds: string[]): Promise<MosPair> {
  const search = new URLSearchParams();
  for (const datasetId of datasetIds) search.append("dataset_id", datasetId);
  return backendRequest<MosPair>(`/mos/pair?${search}`);
}

export function saveMosRating(payload: MosRatingRequest): Promise<MosRating> {
  return backendRequest<MosRating>("/mos/ratings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchMosRatings(datasetIds: string[], limit: number, offset: number): Promise<MosRatingPage> {
  const search = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  for (const datasetId of datasetIds) search.append("dataset_id", datasetId);
  return backendRequest<MosRatingPage>(`/mos/ratings?${search}`);
}

export function updateMosRating(id: string, payload: MosRatingUpdateRequest): Promise<MosHistoryItem> {
  return backendRequest<MosHistoryItem>(`/mos/ratings/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteMosRating(id: string): Promise<void> {
  const response = await backendFetch(`/mos/ratings/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}
