import type { WorkflowGraph, WorkflowSchema } from "./types";

export type WorkflowTemplate = {
  id: string;
  name: string;
  description: string;
  build: (schema: WorkflowSchema) => WorkflowGraph;
};

type WorkflowEdge = WorkflowGraph["edges"][number];

function node(schema: WorkflowSchema, id: string, type: string, x: number, y: number, params: object = {}) {
  const info = schema.nodes[type];
  if (!info) throw new Error(`Unknown node type: ${type}`);
  return {
    id,
    type,
    x,
    y,
    params: { ...structuredClone(info.settings_defaults), ...params },
    runtime: structuredClone(info.runtime_defaults),
  };
}

function edge(source_node: string, source_port: string, target_node: string, target_port: string): WorkflowEdge {
  return { source_node, source_port, target_node, target_port };
}

function allAudioSource(schema: WorkflowSchema, y: number, limit = 100) {
  return node(schema, "source", "AllAudioSource", 64, y, { include_virtual: false, limit });
}

export function transcriptTemplate(schema: WorkflowSchema): WorkflowGraph {
  return transcriptionToSegments(schema);
}

function transcriptionToSegments(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      allAudioSource(schema, 260),
      node(schema, "load_audio", "LoadAudio", 330, 260, { sample_rate: 24000, channels: 1 }),
      node(schema, "whisper", "WhisperTranscribe", 600, 260, { language: "auto", batch_size: 16 }),
      node(schema, "segments", "TranscriptToSegments", 870, 260),
      node(schema, "save_segments", "SaveAudioSegments", 1140, 260),
    ],
    edges: [
      edge("source", "audio_ref", "load_audio", "audio_ref"),
      edge("source", "audio_ref", "save_segments", "audio_ref"),
      edge("load_audio", "audio", "whisper", "audio"),
      edge("whisper", "transcript", "segments", "transcript"),
      edge("segments", "segment_group", "save_segments", "segment_group"),
    ],
  };
}

function normalizeAndUpdate(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      allAudioSource(schema, 260),
      node(schema, "load_audio", "LoadAudio", 330, 260, { sample_rate: 24000, channels: 1 }),
      node(schema, "normalize", "NormalizeLoudness", 600, 260, {
        target_lufs: -23,
        target_rms_db: -20,
        silence_threshold_db: -40,
        padding_ms: 120,
        prevent_clipping: true,
        peak_cap_percent: 95,
      }),
      node(schema, "update_audio", "UpdateAudioRecordBytes", 870, 260),
    ],
    edges: [
      edge("source", "audio_ref", "load_audio", "audio_ref"),
      edge("source", "audio_ref", "update_audio", "audio_ref"),
      edge("load_audio", "audio", "normalize", "audio"),
      edge("normalize", "audio", "update_audio", "audio"),
    ],
  };
}

function denoiseNormalizeUpdate(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      allAudioSource(schema, 180),
      node(schema, "load_audio", "LoadAudio", 330, 180, { sample_rate: 24000, channels: 1 }),
      node(schema, "denoise", "DeepFilterNetDenoise", 600, 180, { model: "deepfilternet3", strength: 0.8 }),
      node(schema, "normalize", "NormalizeLoudness", 870, 180, {
        target_lufs: -23,
        target_rms_db: -20,
        silence_threshold_db: -40,
        padding_ms: 120,
        prevent_clipping: true,
        peak_cap_percent: 95,
      }),
      node(schema, "update_audio", "UpdateAudioRecordBytes", 1140, 180),
    ],
    edges: [
      edge("source", "audio_ref", "load_audio", "audio_ref"),
      edge("source", "audio_ref", "update_audio", "audio_ref"),
      edge("load_audio", "audio", "denoise", "audio"),
      edge("denoise", "audio", "normalize", "audio"),
      edge("normalize", "audio", "update_audio", "audio"),
    ],
  };
}

function phonemizeSegments(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      allAudioSource(schema, 260),
      node(schema, "load_segments", "LoadAudioSegments", 330, 260),
      node(schema, "phonemize", "PhonemizeSegments", 600, 260, { language: "en-us", mode: "fill", tie: true, workers: 4, threads: 2 }),
      node(schema, "save_segments", "SaveAudioSegments", 870, 260),
    ],
    edges: [
      edge("source", "audio_ref", "load_segments", "audio_ref"),
      edge("source", "audio_ref", "save_segments", "audio_ref"),
      edge("load_segments", "segment_group", "phonemize", "segment_group"),
      edge("phonemize", "segment_group", "save_segments", "segment_group"),
    ],
  };
}

function splitAudio(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      allAudioSource(schema, 260),
      node(schema, "load_segments", "LoadAudioSegments", 330, 260),
      node(schema, "plan_groups", "PlanSegmentGroups", 600, 260, { mode: "create_new", max_merged_duration_seconds: 12 }),
      node(schema, "extract_audio", "ExtractSegmentGroupAudio", 870, 260),
      node(schema, "persist_split", "PersistSplitAudioRecords", 1140, 260, { mode: "create_new", virtual: false }),
    ],
    edges: [
      edge("source", "audio_ref", "load_segments", "audio_ref"),
      edge("load_segments", "segment_group", "plan_groups", "segment_group"),
      edge("plan_groups", "segment_group", "extract_audio", "segment_group"),
      edge("extract_audio", "audio", "persist_split", "audio"),
    ],
  };
}

function datasetStatistics(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      allAudioSource(schema, 260),
      node(schema, "load_audio", "LoadAudio", 330, 260, { sample_rate: 24000, channels: 1 }),
      node(schema, "features", "AnalyzeAudioFeatures", 600, 260, { histogram_bins: 50, silence_threshold_db: -40, hop_length: 512 }),
      node(schema, "aggregate", "AggregateDatasetStatistics", 870, 260, { histogram_bins: 50, silence_threshold_db: -40 }),
      node(schema, "save_statistics", "SaveStatisticsEntry", 1140, 260, { name: "dataset_statistics", metadata: { source: "workflow_template" } }),
    ],
    edges: [
      edge("source", "audio_ref", "load_audio", "audio_ref"),
      edge("load_audio", "audio", "features", "audio"),
      edge("features", "features", "aggregate", "feature_records"),
      edge("aggregate", "statistics", "save_statistics", "statistics"),
    ],
  };
}

export const WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  {
    id: "transcription_to_segments",
    name: "Transcription to segments",
    description: "Transcribe audio, convert ASR spans to segment records, and update audio records.",
    build: transcriptionToSegments,
  },
  {
    id: "normalize_update_records",
    name: "Normalize + update records",
    description: "Normalize loudness and write the normalized bytes back to each source audio record.",
    build: normalizeAndUpdate,
  },
  {
    id: "denoise_normalize_update",
    name: "Denoise, normalize + update",
    description: "Denoise, normalize, and write enhanced bytes back to each source audio record.",
    build: denoiseNormalizeUpdate,
  },
  {
    id: "phonemize_segments",
    name: "Phonemize segments",
    description: "Load saved segment records, fill missing phonemes, and persist the updated segments.",
    build: phonemizeSegments,
  },
  {
    id: "split_audio",
    name: "Split audio",
    description: "Plan segment groups, extract grouped audio, and create persisted split audio records.",
    build: splitAudio,
  },
  {
    id: "dataset_statistics",
    name: "Dataset statistics",
    description: "Analyze audio features, aggregate dataset statistics, and save a statistics entry.",
    build: datasetStatistics,
  },
];
