import { PHON, SENTENCES, SPEAKERS, pick, rng } from "./constants";
import type {
  AudioFile,
  AudioStatus,
  Checkpoint,
  Dataset,
  Segment,
  StatEntry,
  StyleRef,
  Voice,
} from "./types";

/**
 * The library is addressed as ~5M rows. Rows are generated on demand by index
 * (never materialized into an array) so a virtualized table can scroll the whole
 * set without holding it in memory. A real backend would page these server-side.
 */
export const AUDIO_COUNT = 5_000_000;

const FILE_NAMES = ["session", "take", "vo", "narration", "line", "clip", "read", "dialog"];
const SAMPLE_RATES = [22050, 24000, 44100];
const DATASET_IDS = ["ds_vox", "ds_narr", "ds_pod"];

/** Deterministic audio row for a given index — stable across scrolls. */
export function getAudioRow(index: number): AudioFile {
  const r = rng(index + 1);
  // Every 40th file is a long recording (~40–66 min, hundreds of segments) so
  // the editor's windowed timeline is exercised at scale.
  const isLong = index % 40 === 0;
  const dur = isLong ? 2400 + rng(index * 3 + 2) * 1600 : 2.4 + rng(index * 3 + 2) * 46;
  const status: AudioStatus[] = [];
  if (isLong || r > 0.18) status.push("transcribed");
  if (rng(index * 5 + 1) > 0.4) status.push("normalized");
  if (rng(index * 7 + 3) > 0.7) status.push("denoised");
  if (rng(index * 11 + 5) > 0.86) status.push("flagged");
  const nseg = Math.max(1, Math.round(dur / 4 + rng(index * 2) * 3));
  return {
    id: `aud_${index}`,
    name: `${pick(FILE_NAMES, index)}_${String(101 + (index % 8999)).padStart(4, "0")}.wav`,
    speaker: pick(SPEAKERS, index),
    dur,
    sr: pick(SAMPLE_RATES, index),
    status: status.length ? status : ["raw"],
    segments: status.includes("transcribed") ? nseg : 0,
    updated: Date.now() - Math.round(rng(index * 13 + 9) * 86400000 * 30),
    sizeMb: (dur * 1.4).toFixed(1),
    dataset: rng(index * 17 + 4) > 0.32 ? pick(DATASET_IDS, index) : null,
  };
}

/**
 * Reconstruct a file's segments (mock — derived from the file, not stored).
 * Diarization/split can produce OVERLAPPING segments (two speakers talking over
 * each other), so some segments deliberately overlap their neighbour — the
 * editor timeline lays those out in separate lanes.
 */
export function fileSegments(file: AudioFile): Segment[] {
  if (!file.segments) return [];
  const out: Segment[] = [];
  const seed = Number(file.id.replace(/\D/g, "")) || 1;
  const stride = file.dur / file.segments;
  let t = 0;
  for (let i = 0; i < file.segments; i++) {
    const len = stride * (0.7 + rng(seed + i * 9 + 1) * 0.6);
    // ~25% of segments start before the previous one ended — an overlap.
    const overlap = i > 0 && rng(seed + i * 3 + 2) > 0.75;
    const start = overlap ? Math.max(0, t - stride * (0.3 + rng(seed + i * 5) * 0.4)) : t;
    const end = Math.min(file.dur, start + len);
    out.push({
      id: `${file.id}_s${i}`,
      start,
      end,
      text: pick(SENTENCES, i),
      phon: pick(PHON, i),
      speaker: pick(SPEAKERS, seed + i),
    });
    t = end + stride * 0.06;
  }
  return out;
}

export const VOICE_COUNT = 1_247;

const FIRST = ["Maya", "Theo", "Aria", "Sam", "Noah", "Liam", "Emma", "Ava", "Ethan", "Mia", "Zoe", "Kai", "Ivy", "Omar", "Nina", "Leo"];
const LAST = ["Chen", "Park", "Russo", "Okafor", "Vance", "Hale", "Mori", "Singh", "Frost", "Cruz", "Webb", "Diaz", "Reed", "Khan"];

function voiceDatasets(i: number): string[] {
  const out = DATASET_IDS.filter((_, k) => rng(i * 5 + k + 1) > 0.55);
  return out.length ? out : [pick(DATASET_IDS, i)];
}

/** Deterministic voice row by index. */
export function getVoiceRow(index: number): Voice {
  if (index < SPEAKERS.length)
    return {
      id: `v_${index}`,
      name: pick(SPEAKERS, index),
      segments: 40 + Math.round(rng(index * 3 + 2) * 220),
      datasets: voiceDatasets(index),
    };
  const i = index - SPEAKERS.length;
  return {
    id: `v_g${i}`,
    name: `${pick(FIRST, i * 7)} ${pick(LAST, i * 13)} ${100 + i}`,
    segments: Math.round(rng(i + 9) * 420),
    datasets: voiceDatasets(i + 30),
  };
}

export function seedDatasets(): Dataset[] {
  return [
    { id: "ds_vox", name: "vox_studio_v3", files: 14 },
    { id: "ds_narr", name: "narration_set", files: 6 },
    { id: "ds_pod", name: "podcast_clean", files: 4 },
  ];
}

export function seedCheckpoints(): Checkpoint[] {
  const now = Date.now();
  return [
    { id: "ck_1", name: "vox_studio_v3_ep50", type: "styletts2", job: "job_8f2a", spkMode: "multi", decoder: "hifigan", symbols: 178, created: now - 3600000 * 6 },
    { id: "ck_2", name: "vox_studio_v3_ep35", type: "styletts2", job: "job_8f2a", spkMode: "multi", decoder: "hifigan", symbols: 178, created: now - 3600000 * 30 },
    { id: "ck_3", name: "narration_ft_ep40", type: "styletts2", job: "job_3a1c", spkMode: "single", decoder: "istftnet", symbols: 178, created: now - 86400000 * 3 },
    { id: "ck_4", name: "asr_aligner_v2", type: "asr", job: "job_77ab", spkMode: "—", decoder: "—", symbols: 178, created: now - 86400000 * 5 },
    { id: "ck_5", name: "f0_jdc_v2", type: "f0", job: "job_77ab", spkMode: "—", decoder: "—", symbols: 0, created: now - 86400000 * 5 },
    { id: "ck_6", name: "plbert_step1M", type: "plbert", job: "—", spkMode: "—", decoder: "—", symbols: 178, created: now - 86400000 * 20 },
  ];
}

export function seedStatEntries(): StatEntry[] {
  const now = Date.now();
  return [
    { id: "st_2041", files: 512, created: now - 3600000 * 5 },
    { id: "st_1990", files: 340, created: now - 86400000 * 2 },
    { id: "st_1873", files: 128, created: now - 86400000 * 9 },
  ];
}

export function seedStyleRefs(): StyleRef[] {
  return [
    { id: "sr_1", name: "maya_calm_ref.wav", voice: "Maya Chen" },
    { id: "sr_2", name: "theo_news_ref.wav", voice: "Theo Park" },
    { id: "sr_3", name: "aria_warm_ref.wav", voice: "Aria Russo" },
  ];
}
