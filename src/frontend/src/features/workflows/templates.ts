import type { WorkflowGraph, WorkflowSchema } from "./types";

export type WorkflowTemplate = {
  id: string;
  name: string;
  description: string;
  build: (schema: WorkflowSchema) => WorkflowGraph;
};

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
  return basicTranscript(schema);
}

function basicTranscript(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      node(schema, "source", "AllAudioSource", 64, 220, { include_virtual: false, limit: 100 }),
      node(schema, "load_audio", "LoadAudio", 340, 220, { sample_rate: 24000, channels: 1 }),
      node(schema, "whisper", "WhisperTranscribe", 620, 220, { language: "auto", batch_size: 16 }),
      node(schema, "save_transcript", "SaveTranscript", 900, 220),
    ],
    edges: [
      { source_node: "source", source_port: "audio_ref", target_node: "load_audio", target_port: "audio_ref" },
      { source_node: "load_audio", source_port: "audio", target_node: "whisper", target_port: "audio" },
      { source_node: "whisper", source_port: "transcript", target_node: "save_transcript", target_port: "transcript" },
    ],
  };
}

function multiAsrReview(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      node(schema, "source", "AllAudioSource", 64, 260, { include_virtual: false, limit: 100 }),
      node(schema, "load_audio", "LoadAudio", 330, 260, { sample_rate: 24000, channels: 1 }),
      node(schema, "whisper", "WhisperTranscribe", 610, 80, { language: "auto", batch_size: 16 }),
      node(schema, "canary", "CanaryTranscribe", 610, 260, { language: "auto", batch_size: 12 }),
      node(schema, "parakeet", "ParakeetTranscribe", 610, 440, { language: "auto", batch_size: 24 }),
      node(schema, "save_whisper", "SaveTranscript", 910, 80, { output_subdir: "transcripts/whisper" }),
      node(schema, "save_canary", "SaveTranscript", 910, 260, { output_subdir: "transcripts/canary" }),
      node(schema, "save_parakeet", "SaveTranscript", 910, 440, { output_subdir: "transcripts/parakeet" }),
    ],
    edges: [
      { source_node: "source", source_port: "audio_ref", target_node: "load_audio", target_port: "audio_ref" },
      { source_node: "load_audio", source_port: "audio", target_node: "whisper", target_port: "audio" },
      { source_node: "load_audio", source_port: "audio", target_node: "canary", target_port: "audio" },
      { source_node: "load_audio", source_port: "audio", target_node: "parakeet", target_port: "audio" },
      { source_node: "whisper", source_port: "transcript", target_node: "save_whisper", target_port: "transcript" },
      { source_node: "canary", source_port: "transcript", target_node: "save_canary", target_port: "transcript" },
      { source_node: "parakeet", source_port: "transcript", target_node: "save_parakeet", target_port: "transcript" },
    ],
  };
}

function segmentationAsr(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      node(schema, "source", "AllAudioSource", 64, 360, { include_virtual: false, limit: 100 }),
      node(schema, "load_audio", "LoadAudio", 330, 360, { sample_rate: 24000, channels: 1 }),
      node(schema, "vad", "VadDetect", 610, 220, { min_segment_sec: 1.0, max_segment_sec: 12.0, padding_sec: 0.12, max_silence_gap_ms: 400 }),
      node(schema, "cut_segments", "CutAudioBySegments", 610, 400, { fade_ms: 5 }),
      node(schema, "save_segments", "SaveAudioArtifact", 900, 560, { output_subdir: "audio/segments", extension: "json" }),
      node(schema, "whisper", "WhisperTranscribe", 900, 120, { language: "auto", batch_size: 32 }),
      node(schema, "canary", "CanaryTranscribe", 900, 300, { language: "auto", batch_size: 24 }),
      node(schema, "parakeet", "ParakeetTranscribe", 900, 480, { language: "auto", batch_size: 32 }),
      node(schema, "save_whisper", "SaveTranscript", 1200, 120, { output_subdir: "transcripts/segments/whisper" }),
      node(schema, "save_canary", "SaveTranscript", 1200, 300, { output_subdir: "transcripts/segments/canary" }),
      node(schema, "save_parakeet", "SaveTranscript", 1200, 480, { output_subdir: "transcripts/segments/parakeet" }),
    ],
    edges: [
      { source_node: "source", source_port: "audio_ref", target_node: "load_audio", target_port: "audio_ref" },
      { source_node: "load_audio", source_port: "audio", target_node: "vad", target_port: "audio" },
      { source_node: "load_audio", source_port: "audio", target_node: "cut_segments", target_port: "audio" },
      { source_node: "vad", source_port: "audio", target_node: "cut_segments", target_port: "segment" },
      { source_node: "cut_segments", source_port: "audio", target_node: "save_segments", target_port: "audio" },
      { source_node: "cut_segments", source_port: "audio", target_node: "whisper", target_port: "audio" },
      { source_node: "cut_segments", source_port: "audio", target_node: "canary", target_port: "audio" },
      { source_node: "cut_segments", source_port: "audio", target_node: "parakeet", target_port: "audio" },
      { source_node: "whisper", source_port: "transcript", target_node: "save_whisper", target_port: "transcript" },
      { source_node: "canary", source_port: "transcript", target_node: "save_canary", target_port: "transcript" },
      { source_node: "parakeet", source_port: "transcript", target_node: "save_parakeet", target_port: "transcript" },
    ],
  };
}

function enhancementStats(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      node(schema, "source", "AllAudioSource", 64, 260, { include_virtual: false, limit: 100 }),
      node(schema, "load_audio", "LoadAudio", 330, 260, { sample_rate: 24000, channels: 1 }),
      node(schema, "denoise", "DeepFilterNetDenoise", 610, 170, { model: "deepfilternet3", strength: 0.8 }),
      node(schema, "normalize", "NormalizeLoudness", 880, 170, { target_lufs: -23, target_rms_db: -20, silence_threshold_db: -40, padding_ms: 120, prevent_clipping: true, peak_cap_percent: 95 }),
      node(schema, "stats", "CalculateAudioStats", 880, 350, { histogram_bins: 50, silence_threshold_db: -40 }),
      node(schema, "save_audio", "SaveAudioArtifact", 1160, 170, { output_subdir: "audio/enhanced" }),
    ],
    edges: [
      { source_node: "source", source_port: "audio_ref", target_node: "load_audio", target_port: "audio_ref" },
      { source_node: "load_audio", source_port: "audio", target_node: "denoise", target_port: "audio" },
      { source_node: "denoise", source_port: "audio", target_node: "normalize", target_port: "audio" },
      { source_node: "normalize", source_port: "audio", target_node: "stats", target_port: "audio" },
      { source_node: "normalize", source_port: "audio", target_node: "save_audio", target_port: "audio" },
    ],
  };
}

function voicePrep(schema: WorkflowSchema): WorkflowGraph {
  return {
    nodes: [
      node(schema, "source", "AllAudioSource", 64, 360, { include_virtual: false, limit: 250 }),
      node(schema, "load_audio", "LoadAudio", 310, 360, { sample_rate: 24000, channels: 1 }),
      node(schema, "denoise", "DeepFilterNetDenoise", 560, 230, { model: "deepfilternet3", strength: 0.7 }),
      node(schema, "normalize", "NormalizeLoudness", 810, 230, { target_lufs: -23, target_rms_db: -20, silence_threshold_db: -45, padding_ms: 80, prevent_clipping: true, peak_cap_percent: 95 }),
      node(schema, "stats", "CalculateAudioStats", 810, 430, { histogram_bins: 80, silence_threshold_db: -45 }),
      node(schema, "vad", "VadDetect", 1060, 230, { min_segment_sec: 1.2, max_segment_sec: 10.0, padding_sec: 0.08, max_silence_gap_ms: 300 }),
      node(schema, "cut_segments", "CutAudioBySegments", 1060, 430, { fade_ms: 5 }),
      node(schema, "whisper", "WhisperTranscribe", 1330, 260, { language: "auto", batch_size: 32 }),
      node(schema, "phonemize", "PhonemizeTranscript", 1600, 260, { language: "en-us", tie: true, workers: 4, threads: 2 }),
      node(schema, "save_audio", "SaveAudioArtifact", 1330, 500, { output_subdir: "audio/prepared_segments", extension: "json" }),
      node(schema, "save_transcript", "SaveTranscript", 1870, 260, { output_subdir: "transcripts/phonemized" }),
    ],
    edges: [
      { source_node: "source", source_port: "audio_ref", target_node: "load_audio", target_port: "audio_ref" },
      { source_node: "load_audio", source_port: "audio", target_node: "denoise", target_port: "audio" },
      { source_node: "denoise", source_port: "audio", target_node: "normalize", target_port: "audio" },
      { source_node: "normalize", source_port: "audio", target_node: "stats", target_port: "audio" },
      { source_node: "normalize", source_port: "audio", target_node: "vad", target_port: "audio" },
      { source_node: "normalize", source_port: "audio", target_node: "cut_segments", target_port: "audio" },
      { source_node: "vad", source_port: "audio", target_node: "cut_segments", target_port: "segment" },
      { source_node: "cut_segments", source_port: "audio", target_node: "whisper", target_port: "audio" },
      { source_node: "cut_segments", source_port: "audio", target_node: "save_audio", target_port: "audio" },
      { source_node: "whisper", source_port: "transcript", target_node: "phonemize", target_port: "transcript" },
      { source_node: "phonemize", source_port: "transcript", target_node: "save_transcript", target_port: "transcript" },
    ],
  };
}

export const WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  { id: "basic_transcript", name: "Basic transcript", description: "Load audio, transcribe with Whisper, save transcript JSON.", build: basicTranscript },
  { id: "multi_asr_review", name: "Multi-ASR review", description: "Fan out loaded audio through Whisper, Canary, and Parakeet.", build: multiAsrReview },
  { id: "segmentation_asr", name: "Segmented ASR", description: "Detect speech regions, cut segments, transcribe each segment.", build: segmentationAsr },
  { id: "enhancement_stats", name: "Enhance + stats", description: "Denoise, normalize, calculate stats, and save enhanced audio.", build: enhancementStats },
  { id: "voice_prep", name: "Voice prep", description: "Enhance, segment, transcribe, phonemize, and save prepared training artifacts.", build: voicePrep },
];
