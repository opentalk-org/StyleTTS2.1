import type { Job } from "./types";

/** Seed jobs spanning every status the Jobs table must render. */
export function seedJobs(): Job[] {
  const now = Date.now();
  return [
    { id: "job_8f2a", type: "training", label: "StyleTTS finetune · vox_studio_v3", status: "running", progress: 42, updated: now - 8000, epoch: 21, step: 4820, machines: 1 },
    { id: "job_3a1c", type: "transcribe", label: "Transcribe · 1,204 files", status: "running", progress: 68, updated: now - 3000, machines: 5 },
    { id: "job_77ab", type: "normalize", label: "Normalize loudness · 320 files", status: "running", progress: 15, updated: now - 92000, stale: true, machines: 4 },
    { id: "job_51de", type: "denoise", label: "Denoise · 48 files", status: "queued", progress: 0, updated: now - 12000, machines: 2 },
    { id: "job_2b90", type: "statistics", label: "Calculate statistics · st_2041", status: "succeeded", progress: 100, updated: now - 3600000 * 5 },
    { id: "job_9c14", type: "split", label: "Split into segments · 96 files", status: "succeeded", progress: 100, updated: now - 3600000 * 9 },
    { id: "job_4f7e", type: "phonemize", label: "Phonemize · 512 files", status: "failed", progress: 63, updated: now - 3600000 * 2, error: "espeak-ng worker crashed on line 3182 (unmapped symbol)." },
  ];
}
