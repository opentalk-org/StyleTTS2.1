import { backendRequest } from "@/app/backend";
import type { AudioSort } from "./store";

export type AudioSegment = {
  id: string;
  start: number;
  end: number;
  text: string;
  phon: string;
  speaker: string;
};

export type AudioFile = {
  id: string;
  name: string;
  speaker: string;
  duration: number;
  sample_rate: number | null;
  byte_length: number;
  size_mb: string;
  segments: number;
  segment_preview: AudioSegment[];
  dataset_ids: string[];
  virtual: boolean;
  metadata: Record<string, unknown>;
  updated_at: string;
};

export type AudioQuery = {
  query: string;
  dataset: string;
  sort: AudioSort;
  limit: number;
  offset: number;
};

export type AudioPage = {
  rows: AudioFile[];
  total: number;
};

export function fetchAudioFiles(params: AudioQuery): Promise<AudioPage> {
  const search = new URLSearchParams({
    query: params.query,
    dataset: params.dataset,
    sort: params.sort,
    limit: String(params.limit),
    offset: String(params.offset),
  });
  return backendRequest<AudioPage>(`/audio-files?${search}`);
}
