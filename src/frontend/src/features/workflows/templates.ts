import type { WorkflowGraph, WorkflowSchema } from "./types";

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

export function transcriptTemplate(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      node(schema, "source", "AllAudioSource", 64, 220, { include_virtual: false, limit: 100 }),
      node(schema, "load_audio", "LoadBucketAudio", 340, 220),
      node(schema, "whisper", "WhisperTranscribe", 620, 220),
      node(schema, "save_transcript", "SaveTranscript", 900, 220),
    ],
    edges: [
      { source_node: "source", source_port: "audio", target_node: "load_audio", target_port: "audio" },
      { source_node: "load_audio", source_port: "audio", target_node: "whisper", target_port: "audio" },
      { source_node: "whisper", source_port: "transcript", target_node: "save_transcript", target_port: "transcript" },
    ],
  };
}
