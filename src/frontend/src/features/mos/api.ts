import { backendRequest } from "@/app/backend";

export type MosAudio = {
  id: string;
  name: string;
  duration: number;
  score: number | null;
  speaker: string;
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

export type MosRating = MosRatingRequest & {
  id: string;
  created_at: string;
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
