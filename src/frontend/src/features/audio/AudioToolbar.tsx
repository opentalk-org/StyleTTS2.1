import { useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import { DATASETS_KEY, useDatasetsQuery } from "@/features/datasets/query";
import { SPEAKER_NAMES } from "@/features/voices/constants";
import { openParamModal, type ParamValues } from "@/shared/feedback/ParamModal";
import { ProgressBar } from "@/shared/feedback/ProgressBar";
import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Select } from "@/shared/ui/Select";
import { type AudioUploadProgress, uploadAudioFiles } from "./api";
import { datasetOptions, sortOptions } from "./logic";
import { AUDIO_FILES_KEY } from "./query";
import { type AudioSort, useAudio } from "./store";

export function AudioToolbar() {
  const { query, dataset, sort, limit, setFilters } = useAudio();
  const { data: datasets = [] } = useDatasetsQuery();
  const queryClient = useQueryClient();
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<AudioUploadProgress | null>(null);

  const submitUpload = async (values: ParamValues) => {
    const files = audioFiles(values.files);
    if (!files.length) {
      showToast("Select audio files first", undefined, "error");
      return;
    }
    setUploading(true);
    setProgress({
      fileName: files[0]!.name,
      fileIndex: 0,
      fileCount: files.length,
      filePercent: 0,
      overallPercent: 0,
      phase: "decoding",
    });
    try {
      const uploaded = await uploadAudioFiles(files, {
        datasetId: String(values.target),
        speaker_id: String(values.speaker_id),
      }, setProgress);
      await queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] });
      await queryClient.invalidateQueries({ queryKey: [DATASETS_KEY] });
      showToast(`Uploaded ${uploaded.length} audio file${uploaded.length === 1 ? "" : "s"}`);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Audio upload failed", undefined, "error");
    } finally {
      setUploading(false);
      setProgress(null);
    }
  };

  const openUpload = () => {
    const target = dataset !== "all" && dataset !== "unassigned" ? dataset : "";
    openParamModal({
      icon: "upload",
      title: "Upload audio",
      submitLabel: "Upload",
      fields: [
        { key: "files", type: "drop", label: "Drop audio files here or click to browse", hint: "WAV, FLAC, MP3, OGG, M4A", accept: "audio/*,.wav,.flac,.mp3,.ogg,.m4a", multiple: true },
        { key: "target", type: "select", label: "Add to dataset", default: target, options: [{ value: "", label: "No dataset" }, ...datasets.map((d) => ({ value: d.id, label: `${d.name} (${d.files})` }))] },
        { key: "speaker_id", type: "select", label: "Assign voice", default: "", options: [{ value: "", label: "None" }, ...SPEAKER_NAMES.map((s) => ({ value: s, label: s }))] },
      ],
      onSubmit: (values) => {
        void submitUpload(values);
      },
    });
  };

  return (
    <div className="mb-3.5">
      <div className="flex flex-wrap items-center gap-2.5">
        <SearchInput
          value={query}
          onChange={(v) => setFilters({ query: v, offset: 0 })}
          placeholder="Search files or speakers…"
        />
        <Select variant="mini" value={dataset} onChange={(v) => setFilters({ dataset: v, offset: 0 })} options={datasetOptions(datasets)} />
        <div className="flex-1" />
        <Select variant="mini" value={sort} onChange={(v) => setFilters({ sort: v as AudioSort, offset: 0 })} options={sortOptions()} />
        <Select
          variant="mini"
          value={String(limit)}
          onChange={(v) => setFilters({ limit: Number(v), offset: 0 })}
          options={[
            { value: "50", label: "50 per page" },
            { value: "100", label: "100 per page" },
            { value: "200", label: "200 per page" },
          ]}
        />
        <Button variant="primary" icon="upload" onClick={openUpload} disabled={uploading}>
          {uploading ? "Uploading..." : "Upload"}
        </Button>
      </div>
      {progress ? <UploadProgress progress={progress} /> : null}
    </div>
  );
}

function UploadProgress({ progress }: { progress: AudioUploadProgress }) {
  const label = progress.phase === "decoding" ? "Preparing" : "Uploading";
  return (
    <div className="mt-2.5 rounded-md border border-line bg-panel px-3 py-2">
      <div className="mb-1.5 flex items-center justify-between gap-3 text-xs text-txt-mute">
        <span className="min-w-0 truncate">
          {label} {progress.fileIndex + 1} of {progress.fileCount}: {progress.fileName}
        </span>
        <span className="font-mono tabular-nums">{Math.round(progress.overallPercent)}%</span>
      </div>
      <ProgressBar value={progress.overallPercent} />
    </div>
  );
}

function audioFiles(value: ParamValues[string] | undefined): File[] {
  return Array.isArray(value) ? value.filter((item): item is File => item instanceof File) : [];
}
