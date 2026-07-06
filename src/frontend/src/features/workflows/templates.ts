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
  return node(schema, "source", "AudioSource", 64, y, { source: "all", include_virtual: false, limit });
}

export function transcriptTemplate(schema: WorkflowSchema): WorkflowGraph {
  return diarizedAsrComparison(schema);
}

function diarizedAsrComparison(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      allAudioSource(schema, 300),
      node(schema, "load_audio", "LoadAudio", 320, 300, { sample_rate: 24000, channels: 1 }),
      node(schema, "sortformer_model", "ModelDownload", 600, 150, { kind: "sortformer", model: "nvidia/diar_sortformer_4spk-v1" }),
      node(schema, "diarize", "SortformerDiarization", 880, 300),
      node(schema, "whisper_model", "ModelDownload", 1160, 80, { kind: "whisper", model: "small" }),
      node(schema, "whisper", "WhisperTranscribe", 1440, 120, { language: "auto" }),
      node(schema, "save_whisper", "SaveAudioSegments", 1720, 120, { mode: "append" }),
      node(schema, "parakeet_model", "ModelDownload", 1160, 300, { kind: "parakeet", model: "nvidia/parakeet-tdt-0.6b-v3" }),
      node(schema, "parakeet", "ParakeetTranscribe", 1440, 300, { batch_size: 16, segment_batch_size: 16 }),
      node(schema, "save_parakeet", "SaveAudioSegments", 1720, 300, { mode: "append" }),
      node(schema, "canary_model", "ModelDownload", 1160, 520, { kind: "canary", model: "nvidia/canary-1b-v2" }),
      node(schema, "canary", "CanaryTranscribe", 1440, 520, { language: "pl", target_language: "pl", batch_size: 16 }),
      node(schema, "save_canary", "SaveAudioSegments", 1720, 520, { mode: "append" }),
    ],
    edges: [
      edge("source", "audio", "load_audio", "audio"),
      edge("sortformer_model", "checkpoint_ref", "diarize", "checkpoint"),
      edge("load_audio", "audio", "diarize", "audio"),
      edge("diarize", "audio", "whisper", "audio"),
      edge("whisper_model", "checkpoint_ref", "whisper", "checkpoint"),
      edge("whisper", "audio", "save_whisper", "audio"),
      edge("diarize", "audio", "parakeet", "audio"),
      edge("parakeet_model", "checkpoint_ref", "parakeet", "checkpoint"),
      edge("parakeet", "audio", "save_parakeet", "audio"),
      edge("diarize", "audio", "canary", "audio"),
      edge("canary_model", "checkpoint_ref", "canary", "checkpoint"),
      edge("canary", "audio", "save_canary", "audio"),
    ],
  };
}

export const WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  {
    id: "sortformer_asr_comparison",
    name: "Sortformer ASR comparison",
    description: "Diarize audio with Sortformer, transcribe each speaker chunk with Whisper, Parakeet, and Canary, then save all three transcription sets.",
    build: diarizedAsrComparison,
  },
];
