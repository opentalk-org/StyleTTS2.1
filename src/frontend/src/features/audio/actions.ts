import type { Dataset } from "@/features/datasets/api";
import { SPEAKER_NAMES } from "@/features/voices/constants";
import { showToast } from "@/shared/feedback/Toast";
import { openParamModal } from "@/shared/feedback/ParamModal";

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
      { key: "speaker", type: "select", label: "Assign voice", default: "", options: [{ value: "", label: "None" }, ...SPEAKER_NAMES.map((s) => ({ value: s, label: s }))] },
    ],
    onSubmit: () => {
      showToast("Upload queued");
    },
  });
}

export function addDatasetAction(datasets: Dataset[], onAdd: (datasetId: string) => void) {
  const options = datasetSelect(datasets);
  openParamModal({
    icon: "database",
    title: "Add to dataset",
    submitLabel: "Add",
    fields: [
      { key: "target", type: "select", label: "Dataset", default: options[0]?.value ?? "", options },
    ],
    onSubmit: (values) => onAdd(String(values.target)),
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
      { key: "voice", type: "select", label: "Voice", default: "", options: [{ value: "", label: "None" }, ...SPEAKER_NAMES.map((s) => ({ value: s, label: s }))] },
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
