import { backendFetch, backendRequest } from "@/app/backend";

export type Histogram = { edges: number[]; counts: number[] };
export type Pair = [string, number];
export type BigramMatrix = { labels: string[]; matrix: number[][] };
export type TextWarning = { audio_file_id: string; name: string; char_count: number; reason: string };
export type PerFileText = { audio_file_id: string; text: string; phon: string };

export type StatisticsPayload = {
  version: number;
  params: Record<string, number>;
  audio_file_ids: string[];
  file_count: number;
  segment_count: number;
  speaker_count: number;
  total_duration_seconds: number;
  mean_duration_seconds: number;
  median_duration_seconds: number;
  total_char_count: number;
  duplicate_segments_collapsed: number;
  phonemes_available: boolean;
  text_length_warnings: TextWarning[];
  duration_seconds_histogram: Histogram;
  char_count_per_file_histogram: Histogram;
  phoneme_count_per_file_histogram: Histogram;
  char_unigram_counts: Pair[];
  phoneme_unigram_counts: Pair[];
  char_bigram_matrix: BigramMatrix;
  phoneme_bigram_matrix: BigramMatrix;
  char_trigram_top10: Pair[];
  char_trigram_bottom10: Pair[];
  phoneme_trigram_top10: Pair[];
  phoneme_trigram_bottom10: Pair[];
  speaker_duration_seconds: Pair[];
  speaker_char_count: Pair[];
  speaker_phoneme_count: Pair[];
  rms_db_histogram: Histogram;
  frame_value_min_histogram: Histogram;
  frame_value_max_histogram: Histogram;
  frame_value_mean_histogram: Histogram;
  mean_rms_nonsilent_db_per_file_histogram: Histogram;
  sample_rms_nonsilent_db_per_file_histogram: Histogram;
  clipped_audio_file_count: number;
  clipped_sample_count_top: number;
  clipped_sample_count_bottom: number;
  silence_ratio_histogram: Histogram;
  silence_rms_db_histogram: Histogram;
  per_file_text: PerFileText[];
};

export type StatisticsSummary = {
  id: string;
  name: string;
  dataset_id: string | null;
  file_count: number;
  created_at: string;
};

export type StatisticsEntry = {
  id: string;
  name: string;
  dataset_id: string | null;
  payload: StatisticsPayload;
  metadata: Record<string, unknown>;
  created_at: string;
};

export function fetchStatisticsEntries(): Promise<StatisticsSummary[]> {
  return backendRequest<StatisticsSummary[]>("/statistics");
}

export function fetchStatisticsEntry(id: string): Promise<StatisticsEntry> {
  return backendRequest<StatisticsEntry>(`/statistics/${id}`);
}

export async function deleteStatisticsEntry(id: string): Promise<void> {
  const response = await backendFetch(`/statistics/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status}`);
  }
}
