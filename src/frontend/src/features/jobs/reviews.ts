import { backendResourceUrl } from "@/app/backend";
import type { Tone } from "@/shared/ui/Badge";
import type { AudioSegmentReviewMedia, ReviewState, ReviewTone } from "./api";

const REVIEW_TONES: Record<ReviewTone, Tone> = {
  neutral: "gray",
  success: "emerald",
  warning: "amber",
  danger: "red",
};

const STATE_TONES: Record<ReviewState, Tone> = {
  pending: "amber",
  approved: "emerald",
  rejected: "red",
};

export function reviewTone(tone: ReviewTone): Tone {
  return REVIEW_TONES[tone];
}

export function reviewStateTone(state: ReviewState): Tone {
  return STATE_TONES[state];
}

export function reviewMediaUrl(media: AudioSegmentReviewMedia): string {
  return backendResourceUrl(`/audio-files/${encodeURIComponent(media.audio_file_id)}/content`);
}
