import { formatAudioScore, parseAudioScore } from "@/features/audio/AudioScoreInput";
import type { MosPair, MosRatingRequest, MosRatingUpdateRequest } from "./api";

export function pairScoreDraft(score: number | null): string {
  return formatAudioScore(score);
}

export function mosRatingRequest(
  pair: MosPair,
  scoreADraft: string,
  scoreBDraft: string,
  preferredAudioId: string | null,
): MosRatingRequest {
  const scoreA = parseAudioScore(scoreADraft);
  const scoreB = parseAudioScore(scoreBDraft);
  if (scoreA === null || scoreB === null) throw new Error("Enter a numeric score for both audio files");
  if (preferredAudioId !== pair.audio_a.id && preferredAudioId !== pair.audio_b.id) {
    throw new Error("Choose the better audio file");
  }
  return {
    dataset_id: pair.dataset_id,
    audio_a_id: pair.audio_a.id,
    audio_b_id: pair.audio_b.id,
    preferred_audio_id: preferredAudioId,
    score_a: scoreA,
    score_b: scoreB,
  };
}

export function hasCompleteMosScores(scoreADraft: string, scoreBDraft: string): boolean {
  return parseAudioScore(scoreADraft) !== null && parseAudioScore(scoreBDraft) !== null;
}

export function mosRatingUpdateRequest(
  scoreADraft: string,
  scoreBDraft: string,
  preferredAudioId: string,
): MosRatingUpdateRequest {
  const scoreA = parseAudioScore(scoreADraft);
  const scoreB = parseAudioScore(scoreBDraft);
  if (scoreA === null || scoreB === null) throw new Error("Enter a numeric score for both audio files");
  return { preferred_audio_id: preferredAudioId, score_a: scoreA, score_b: scoreB };
}

export function audioSeed(audioId: string): number {
  let seed = 0;
  for (const char of audioId) seed = (seed * 31 + char.charCodeAt(0)) >>> 0;
  return seed;
}
