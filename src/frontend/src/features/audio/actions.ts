import type { Dataset } from "@/features/datasets/api";
import { SPEAKERS } from "@/mock/constants";
import { showToast } from "@/shared/feedback/Toast";
import { openParamModal, type ParamValues } from "@/shared/feedback/ParamModal";

/** How many files an action targets, for job/toast labelling. */
function scope(count: number | undefined): string {
  return count ? `${count} file${count === 1 ? "" : "s"}` : "selection";
}

function datasetSelect(datasets: Dataset[]) {
  return datasets.map((d) => ({ value: d.id, label: `${d.name} (${d.files})` }));
}

export function uploadAction(datasets: Dataset[]) {
  const options = datasetSelect(datasets);
  openParamModal({
    icon: "upload",
    title: "Upload audio",
    submitLabel: "Upload",
    fields: [
      { type: "drop", label: "Drop audio files here or click to browse", hint: "WAV, FLAC, MP3 · up to 2 GB per file" },
      { key: "target", type: "select", label: "Add to dataset", default: options[0]?.value ?? "", options },
      { key: "speaker", type: "select", label: "Assign speaker", default: SPEAKERS[0]!, options: SPEAKERS.map((s) => ({ value: s, label: s })) },
    ],
    onSubmit: () => {
      showToast("Upload queued");
    },
  });
}

export function splitAction(count: number | undefined, datasets: Dataset[]) {
  const options = datasetSelect(datasets);
  openParamModal({
    icon: "scissors",
    title: "Split into segments",
    desc: `Split ${scope(count)} by silence into transcribable segments.`,
    submitLabel: "Split",
    fields: [
      { key: "mode", type: "radio", label: "Mode", default: "new", options: [{ value: "new", label: "Create new" }, { value: "replace", label: "Replace all" }] },
      { key: "target", type: "select", label: "Target dataset", default: options[0]?.value ?? "", options },
      { key: "minlen", type: "number", label: "Min length (s)", default: 1.0, min: 0.1, max: 30, step: 0.1 },
      { key: "maxlen", type: "number", label: "Max length (s)", default: 12.0, min: 1, max: 60, step: 0.5 },
      { key: "minchars", type: "number", label: "Min characters", default: 3, min: 0, max: 200, step: 1 },
      { key: "maxchars", type: "number", label: "Max characters", default: 300, min: 10, max: 1000, step: 10 },
      { key: "usegap", type: "toggle", label: "Limit max silence gap", default: false },
      { key: "maxgap", type: "number", label: "Max gap (ms)", default: 400, min: 50, max: 3000, step: 50, showIf: (v: ParamValues) => Boolean(v.usegap) },
    ],
    onSubmit: () => {
      showToast("Split job queued");
    },
  });
}

export function denoiseAction(count?: number) {
  openParamModal({
    icon: "wand",
    title: "Denoise audio",
    desc: `Denoise ${scope(count)}.`,
    submitLabel: "Denoise",
    fields: [
      { key: "model", type: "select", label: "Denoise model", default: "deepfilternet3", options: [{ value: "deepfilternet3", label: "DeepFilterNet 3" }, { value: "demucs", label: "Demucs (htdemucs)" }, { value: "rnnoise", label: "RNNoise (fast)" }] },
      { key: "strength", type: "number", label: "Strength", default: 0.8, min: 0, max: 1, step: 0.05, hint: "0 = bypass, 1 = maximum suppression." },
      { key: "keep", type: "toggle", label: "Keep original files", default: true },
    ],
    onSubmit: () => {
      showToast("Denoise job queued");
    },
  });
}

export function normalizeAction(count?: number) {
  openParamModal({
    icon: "audio-lines",
    title: "Normalize loudness",
    desc: `Normalize ${scope(count)}.`,
    submitLabel: "Normalize",
    fields: [
      { key: "lufs", type: "number", label: "Target loudness (LUFS)", default: -23, min: -40, max: -6, step: 0.5, hint: "EBU R128 integrated loudness target." },
      { key: "rms", type: "number", label: "Target RMS (dB)", default: -20, min: -40, max: 0, step: 0.5 },
      { key: "silence", type: "number", label: "Silence threshold (dB)", default: -40, min: -80, max: 0, step: 1 },
      { key: "pad", type: "number", label: "Padding silence (ms)", default: 120, min: 0, max: 1000, step: 10 },
      { key: "clip", type: "toggle", label: "Prevent peak clipping", default: true },
      { key: "peak", type: "number", label: "Peak cap (%)", default: 95, min: 50, max: 100, step: 1, showIf: (v: ParamValues) => Boolean(v.clip) },
    ],
    onSubmit: () => {
      showToast("Normalize job queued");
    },
  });
}

export function transcribeAction(count?: number) {
  openParamModal({
    icon: "file-audio",
    title: "Transcribe audio",
    desc: `Transcribe ${scope(count)}.`,
    submitLabel: "Transcribe",
    fields: [
      { key: "scope", type: "radio", label: "Scope", default: "segment", options: [{ value: "file", label: "Full file" }, { value: "segment", label: "Per segment" }] },
      { key: "engine", type: "select", label: "ASR engine", default: "whisper-large-v3", options: [{ value: "whisper-large-v3", label: "Whisper large-v3" }, { value: "whisper-medium", label: "Whisper medium" }, { value: "parakeet", label: "Parakeet RNNT 1.1B" }], hint: "Larger models are slower but more accurate." },
      { key: "mode", type: "radio", label: "Write mode", default: "replace", options: [{ value: "replace", label: "Replace existing" }, { value: "add", label: "Only fill empty" }] },
      { key: "batch", type: "number", label: "Batch size", default: 16, min: 1, max: 128, step: 1 },
    ],
    onSubmit: () => {
      showToast("Transcribe job queued");
    },
  });
}

export function phonemizeAction(count?: number) {
  openParamModal({
    icon: "sliders",
    title: "Phonemize transcripts",
    desc: `Phonemize ${scope(count)}.`,
    submitLabel: "Phonemize",
    fields: [
      { key: "mode", type: "radio", label: "Mode", default: "fill", options: [{ value: "fill", label: "Fill empty" }, { value: "replace", label: "Replace all" }] },
      { key: "lang", type: "select", label: "Language", default: "en-us", options: [{ value: "en-us", label: "English (US)" }, { value: "en-gb", label: "English (UK)" }, { value: "es", label: "Spanish" }, { value: "de", label: "German" }, { value: "fr", label: "French" }, { value: "ja", label: "Japanese" }] },
      { key: "tie", type: "toggle", label: "Use tie-bars (͡) for affricates", default: true },
      { key: "workers", type: "number", label: "Worker processes", default: 4, min: 1, max: 32, step: 1 },
      { key: "threads", type: "number", label: "Threads per worker", default: 2, min: 1, max: 16, step: 1 },
    ],
    onSubmit: () => {
      showToast("Phonemize job queued");
    },
  });
}

export function calculateStatisticsAction(count?: number) {
  openParamModal({
    icon: "bar-chart",
    title: "Calculate statistics",
    desc: `Compute duration and loudness distributions over ${scope(count)}.`,
    submitLabel: "Calculate",
    fields: [
      { key: "bins", type: "number", label: "Histogram bins", default: 50, min: 10, max: 200, step: 5 },
      { key: "silence", type: "number", label: "Silence threshold (dB)", default: -40, min: -80, max: 0, step: 1 },
    ],
    onSubmit: () => {
      showToast("Statistics job queued");
    },
  });
}

export function addDatasetAction(count: number | undefined, datasets: Dataset[]) {
  const options = datasetSelect(datasets);
  openParamModal({
    icon: "database",
    title: "Add to dataset",
    submitLabel: "Add",
    fields: [
      { key: "target", type: "select", label: "Dataset", default: options[0]?.value ?? "", options },
    ],
    onSubmit: () => showToast(`Added ${scope(count)} to dataset`),
  });
}

export function removeDatasetAction(count: number | undefined, datasets: Dataset[]) {
  const options = datasetSelect(datasets);
  openParamModal({
    icon: "database",
    title: "Remove from dataset",
    submitLabel: "Remove",
    fields: [
      { key: "target", type: "select", label: "Dataset", default: options[0]?.value ?? "", options },
    ],
    onSubmit: () => showToast(`Removed ${scope(count)} from dataset`, undefined, "error"),
  });
}

export function assignVoiceAction(count?: number) {
  openParamModal({
    icon: "mic",
    title: "Assign voice to segments",
    submitLabel: "Assign",
    fields: [
      { key: "voice", type: "select", label: "Voice", default: SPEAKERS[0]!, options: SPEAKERS.map((s) => ({ value: s, label: s })) },
    ],
    onSubmit: () => showToast(`Assigned voice to ${scope(count)}`),
  });
}

export function removeSegmentsAction(count?: number) {
  openParamModal({
    icon: "trash",
    title: "Remove all segments",
    danger: true,
    submitLabel: "Remove segments",
    fields: [
      { type: "info", label: `This deletes every segment on ${scope(count)}. The audio files themselves are kept.` },
    ],
    onSubmit: () => showToast(`Removed segments from ${scope(count)}`, undefined, "error"),
  });
}

export function deleteFilesAction(count?: number) {
  openParamModal({
    icon: "trash",
    title: "Delete files",
    danger: true,
    submitLabel: "Delete",
    fields: [
      { type: "info", label: `Permanently delete ${scope(count)} and all of their segments. This cannot be undone.` },
    ],
    onSubmit: () => showToast(`Deleted ${scope(count)}`, undefined, "error"),
  });
}
