import type { SchemaValues } from "@/shared/schema-form/types";
import type { Option } from "@/shared/ui/Select";

import type { FileAsset } from "../assets/api";
import type { Checkpoint } from "../checkpoints/api";
import type { Dataset } from "../datasets/api";
import type { WorkflowEdge, WorkflowGraph, WorkflowNode, WorkflowSchema } from "../workflows/types";
import type { TrainTab } from "./store";

export type OodSetValue = { id: string; name: string; line_count: number };

export type TrainingNodeIds = {
  dataset: string;
  checkpoint: string;
  training: string;
  assets?: string;
  alphabet?: string;
  oodSets?: string;
};

export type TrainingWorkflowSpec = {
  value: TrainTab;
  label: string;
  nodes: { id: string; type: string; x: number; y: number }[];
  edges: WorkflowEdge[];
  ids: TrainingNodeIds;
};

const STYLE_TTS_NODES = [
  { id: "run", type: "TrainingRunInput", x: -160, y: 620 },
  { id: "dataset", type: "SelectTrainingDataset", x: 64, y: 220 },
  { id: "audio_ids", type: "ListDatasetAudioIds", x: 330, y: 220 },
  { id: "base_checkpoint", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "prefetch_checkpoint", type: "PrefetchCheckpoint", x: 330, y: 420 },
  { id: "assets", type: "SelectTrainingAssets", x: 64, y: 620 },
  { id: "prefetch_assets", type: "PrefetchTrainingAssets", x: 330, y: 620 },
  { id: "alphabet", type: "PhonemeAlphabet", x: 64, y: 820 },
  { id: "ood_sets", type: "SelectOodTextSets", x: 64, y: 1020 },
  { id: "prefetch_ood_sets", type: "PrefetchOodTextSets", x: 330, y: 1020 },
  { id: "styletts", type: "StyleTtsFinetune", x: 620, y: 620 },
];

const STYLE_TTS_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "dataset", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "base_checkpoint", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "assets", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "alphabet", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "ood_sets", target_port: "run" },
  { source_node: "dataset", source_port: "dataset_ref", target_node: "audio_ids", target_port: "dataset_ref" },
  { source_node: "audio_ids", source_port: "audio_file_ids", target_node: "styletts", target_port: "audio_file_ids" },
  { source_node: "base_checkpoint", source_port: "checkpoint_ref", target_node: "prefetch_checkpoint", target_port: "checkpoint_ref" },
  { source_node: "prefetch_checkpoint", source_port: "checkpoint", target_node: "styletts", target_port: "base_checkpoint" },
  { source_node: "assets", source_port: "asset_refs", target_node: "prefetch_assets", target_port: "asset_refs" },
  { source_node: "prefetch_assets", source_port: "assets", target_node: "styletts", target_port: "pretrained_assets" },
  { source_node: "alphabet", source_port: "phoneme_alphabet", target_node: "styletts", target_port: "phoneme_alphabet" },
  { source_node: "ood_sets", source_port: "ood_text_set_refs", target_node: "prefetch_ood_sets", target_port: "ood_text_set_refs" },
  { source_node: "prefetch_ood_sets", source_port: "ood_text_sets", target_node: "styletts", target_port: "ood_text_sets" },
];

const F0_NODES = [
  { id: "run", type: "TrainingRunInput", x: -160, y: 320 },
  { id: "dataset", type: "SelectTrainingDataset", x: 64, y: 220 },
  { id: "audio_ids", type: "ListDatasetAudioIds", x: 330, y: 220 },
  { id: "pretrained", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "prefetch_checkpoint", type: "PrefetchCheckpoint", x: 330, y: 420 },
  { id: "f0", type: "F0ModelTraining", x: 620, y: 320 },
];

const F0_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "dataset", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "pretrained", target_port: "run" },
  { source_node: "dataset", source_port: "dataset_ref", target_node: "audio_ids", target_port: "dataset_ref" },
  { source_node: "audio_ids", source_port: "audio_file_ids", target_node: "f0", target_port: "audio_file_ids" },
  { source_node: "pretrained", source_port: "checkpoint_ref", target_node: "prefetch_checkpoint", target_port: "checkpoint_ref" },
  { source_node: "prefetch_checkpoint", source_port: "checkpoint", target_node: "f0", target_port: "pretrained_checkpoint" },
];

const ASR_NODES = [
  { id: "run", type: "TrainingRunInput", x: -160, y: 420 },
  { id: "dataset", type: "SelectTrainingDataset", x: 64, y: 220 },
  { id: "audio_ids", type: "ListDatasetAudioIds", x: 330, y: 220 },
  { id: "pretrained", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "prefetch_checkpoint", type: "PrefetchCheckpoint", x: 330, y: 420 },
  { id: "alphabet", type: "PhonemeAlphabet", x: 64, y: 620 },
  { id: "asr", type: "AsrModelTraining", x: 620, y: 420 },
];

const ASR_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "dataset", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "pretrained", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "alphabet", target_port: "run" },
  { source_node: "dataset", source_port: "dataset_ref", target_node: "audio_ids", target_port: "dataset_ref" },
  { source_node: "audio_ids", source_port: "audio_file_ids", target_node: "asr", target_port: "audio_file_ids" },
  { source_node: "pretrained", source_port: "checkpoint_ref", target_node: "prefetch_checkpoint", target_port: "checkpoint_ref" },
  { source_node: "prefetch_checkpoint", source_port: "checkpoint", target_node: "asr", target_port: "pretrained_checkpoint" },
  { source_node: "alphabet", source_port: "phoneme_alphabet", target_node: "asr", target_port: "phoneme_alphabet" },
];

export const TRAINING_WORKFLOWS: Record<TrainTab, TrainingWorkflowSpec> = {
  styletts: {
    value: "styletts",
    label: "StyleTTS finetune",
    nodes: STYLE_TTS_NODES,
    edges: STYLE_TTS_EDGES,
    ids: {
      dataset: "dataset",
      checkpoint: "base_checkpoint",
      assets: "assets",
      alphabet: "alphabet",
      oodSets: "ood_sets",
      training: "styletts",
    },
  },
  f0: {
    value: "f0",
    label: "F0 model",
    nodes: F0_NODES,
    edges: F0_EDGES,
    ids: { dataset: "dataset", checkpoint: "pretrained", training: "f0" },
  },
  asr: {
    value: "asr",
    label: "ASR model",
    nodes: ASR_NODES,
    edges: ASR_EDGES,
    ids: { dataset: "dataset", checkpoint: "pretrained", alphabet: "alphabet", training: "asr" },
  },
};

export const TRAINING_OPTIONS: Option[] = Object.values(TRAINING_WORKFLOWS).map((workflow) => ({
  value: workflow.value,
  label: workflow.label,
}));

export function createTrainingGraph(schema: WorkflowSchema, spec: TrainingWorkflowSpec): WorkflowGraph {
  assertTrainingNodes(schema, spec);
  return {
    nodes: spec.nodes.map((node) => {
      const info = schema.nodes[node.type];
      if (!info) throw new Error(`Training node is not registered: ${node.type}`);
      return {
        id: node.id,
        type: node.type,
        x: node.x,
        y: node.y,
        params: structuredClone(info.settings_defaults),
        runtime: structuredClone(info.runtime_defaults),
      };
    }),
    edges: spec.edges,
  };
}

export function trainingNode(graph: WorkflowGraph, nodeId: string): WorkflowNode {
  const node = graph.nodes.find((item) => item.id === nodeId);
  if (!node) throw new Error(`Training graph is missing node: ${nodeId}`);
  return node;
}

export function updateNodeParams(graph: WorkflowGraph, nodeId: string, params: SchemaValues): WorkflowGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === nodeId ? { ...node, params } : node)),
  };
}

export function assertTrainingNodes(schema: WorkflowSchema, spec: TrainingWorkflowSpec) {
  const missing = spec.nodes.map((node) => node.type).filter((type) => !schema.nodes[type]);
  if (missing.length > 0) {
    throw new Error(`Training workflow nodes are not registered: ${missing.join(", ")}`);
  }
}

export function datasetOptions(datasets: Dataset[]): Option[] {
  return [
    { value: "", label: datasets.length ? "— select training dataset —" : "No datasets available" },
    ...datasets.map((dataset) => ({
      value: dataset.id,
      label: `${dataset.name} (${dataset.files.toLocaleString()} files)`,
    })),
  ];
}

export function checkpointOptions(checkpoints: Checkpoint[], type: string, placeholder: string): Option[] {
  const rows = checkpoints.filter((checkpoint) => checkpoint.type_ === type);
  return [
    { value: "", label: rows.length ? placeholder : `No ${type} checkpoints available` },
    ...rows.map((checkpoint) => ({ value: checkpoint.id, label: checkpoint.name })),
  ];
}

export function fileAssetOptions(assets: FileAsset[], placeholder: string): Option[] {
  return [
    { value: "", label: assets.length ? placeholder : "No files available" },
    ...assets.map((asset) => ({ value: asset.id, label: asset.name })),
  ];
}

export function oodSetValues(assets: FileAsset[]): OodSetValue[] {
  return assets.map((asset) => ({
    id: asset.id,
    name: asset.name,
    line_count: lineCount(asset),
  }));
}

function lineCount(asset: FileAsset): number {
  const value = Number(asset.metadata.line_count);
  return Number.isFinite(value) ? value : 0;
}
