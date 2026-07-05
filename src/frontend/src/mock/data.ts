import { PHON, SENTENCES, SPEAKERS, rng } from "./constants";
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
  const dur = 2.4 + rng(index * 3 + 2) * 46;
  const status: AudioStatus[] = [];
  if (r > 0.18) status.push("transcribed");
  if (rng(index * 5 + 1) > 0.4) status.push("normalized");
  if (rng(index * 7 + 3) > 0.7) status.push("denoised");
  if (rng(index * 11 + 5) > 0.86) status.push("flagged");
  const nseg = Math.max(1, Math.round(dur / 4 + rng(index * 2) * 3));
  return {
    id: `aud_${index}`,
    name: `${FILE_NAMES[index % FILE_NAMES.length]}_${String(101 + (index % 8999)).padStart(4, "0")}.wav`,
    speaker: SPEAKERS[index % SPEAKERS.length],
    dur,
    sr: SAMPLE_RATES[index % SAMPLE_RATES.length],
    status: status.length ? status : ["raw"],
    segments: status.includes("transcribed") ? nseg : 0,
    updated: Date.now() - Math.round(rng(index * 13 + 9) * 86400000 * 30),
    sizeMb: (dur * 1.4).toFixed(1),
    dataset: rng(index * 17 + 4) > 0.32 ? DATASET_IDS[index % 3] : null,
  };
}

/** Reconstruct a file's segments (mock — derived from the file, not stored). */
export function fileSegments(file: AudioFile): Segment[] {
  if (!file.segments) return [];
  const out: Segment[] = [];
  const seed = Number(file.id.replace(/\D/g, "")) || 1;
  let t = 0;
  for (let i = 0; i < file.segments; i++) {
    const len = (file.dur / file.segments) * (0.7 + rng(seed + i * 9 + 1) * 0.6);
    const start = t;
    const end = Math.min(file.dur, i === file.segments - 1 ? file.dur : t + len);
    out.push({
      id: `${file.id}_s${i}`,
      start,
      end,
      text: SENTENCES[i % SENTENCES.length],
      phon: PHON[i % PHON.length],
      speaker: file.speaker,
    });
    t = end + (i < file.segments - 1 ? (file.dur / file.segments) * 0.06 : 0);
  }
  return out;
}

export const VOICE_COUNT = 1_247;

const FIRST = ["Maya", "Theo", "Aria", "Sam", "Noah", "Liam", "Emma", "Ava", "Ethan", "Mia", "Zoe", "Kai", "Ivy", "Omar", "Nina", "Leo"];
const LAST = ["Chen", "Park", "Russo", "Okafor", "Vance", "Hale", "Mori", "Singh", "Frost", "Cruz", "Webb", "Diaz", "Reed", "Khan"];

function voiceDatasets(i: number): string[] {
  const out = DATASET_IDS.filter((_, k) => rng(i * 5 + k + 1) > 0.55);
  return out.length ? out : [DATASET_IDS[i % 3]];
}

/** Deterministic voice row by index. */
export function getVoiceRow(index: number): Voice {
  if (index < SPEAKERS.length)
    return {
      id: `v_${index}`,
      name: SPEAKERS[index],
      segments: 40 + Math.round(rng(index * 3 + 2) * 220),
      datasets: voiceDatasets(index),
    };
  const i = index - SPEAKERS.length;
  return {
    id: `v_g${i}`,
    name: `${FIRST[(i * 7) % FIRST.length]} ${LAST[(i * 13) % LAST.length]} ${100 + i}`,
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
