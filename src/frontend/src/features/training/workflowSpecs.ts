import type { WorkflowEdge } from "../workflows/types";
import type { TrainingWorkflowSpec } from "./logic";
import type { TrainTab } from "./store";

const STYLE_TTS_NODES = [
  { id: "run", type: "TrainingRunInput", x: -160, y: 620 },
  { id: "dataset", type: "SelectTrainingDataset", x: 64, y: 220 },
  { id: "base_checkpoint", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "assets", type: "SelectTrainingAssets", x: 64, y: 620 },
  { id: "alphabet", type: "PhonemeAlphabet", x: 64, y: 820 },
  { id: "styletts", type: "StyleTtsFinetune", x: 620, y: 520 },
];

const STYLE_TTS_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "dataset", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "base_checkpoint", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "assets", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "alphabet", target_port: "run" },
  { source_node: "dataset", source_port: "dataset_ref", target_node: "styletts", target_port: "dataset_ref" },
  { source_node: "alphabet", source_port: "phoneme_alphabet", target_node: "styletts", target_port: "phoneme_alphabet" },
  { source_node: "base_checkpoint", source_port: "checkpoint", target_node: "styletts", target_port: "checkpoint" },
  { source_node: "assets", source_port: "assets", target_node: "styletts", target_port: "assets" },
];

const F0_NODES = [
  { id: "run", type: "TrainingRunInput", x: -160, y: 320 },
  { id: "dataset", type: "SelectTrainingDataset", x: 64, y: 220 },
  { id: "pretrained", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "f0", type: "F0ModelTraining", x: 620, y: 320 },
];

const F0_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "dataset", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "pretrained", target_port: "run" },
  { source_node: "dataset", source_port: "dataset_ref", target_node: "f0", target_port: "dataset_ref" },
  { source_node: "pretrained", source_port: "checkpoint", target_node: "f0", target_port: "checkpoint" },
];

const ASR_NODES = [
  { id: "run", type: "TrainingRunInput", x: -160, y: 420 },
  { id: "dataset", type: "SelectTrainingDataset", x: 64, y: 220 },
  { id: "pretrained", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "alphabet", type: "PhonemeAlphabet", x: 64, y: 620 },
  { id: "asr", type: "AsrModelTraining", x: 620, y: 420 },
];

const ASR_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "dataset", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "pretrained", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "alphabet", target_port: "run" },
  { source_node: "dataset", source_port: "dataset_ref", target_node: "asr", target_port: "dataset_ref" },
  { source_node: "pretrained", source_port: "checkpoint", target_node: "asr", target_port: "checkpoint" },
  { source_node: "alphabet", source_port: "phoneme_alphabet", target_node: "asr", target_port: "phoneme_alphabet" },
];

const MOS_NODES = [
  { id: "run", type: "TrainingRunInput", x: -160, y: 320 },
  { id: "dataset", type: "SelectTrainingDataset", x: 64, y: 220 },
  { id: "base_checkpoint", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "mos", type: "MosModelTraining", x: 620, y: 320 },
];

const MOS_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "dataset", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "base_checkpoint", target_port: "run" },
  { source_node: "dataset", source_port: "dataset_ref", target_node: "mos", target_port: "dataset_ref" },
  { source_node: "base_checkpoint", source_port: "checkpoint", target_node: "mos", target_port: "checkpoint" },
];

export const TRAINING_WORKFLOWS: Record<TrainTab, TrainingWorkflowSpec> = {
  styletts: {
    value: "styletts",
    label: "StyleTTS finetune",
    nodes: STYLE_TTS_NODES,
    edges: STYLE_TTS_EDGES,
    ids: { dataset: "dataset", checkpoint: "base_checkpoint", assets: "assets", alphabet: "alphabet", training: "styletts" },
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
  mos: {
    value: "mos",
    label: "MOS model",
    nodes: MOS_NODES,
    edges: MOS_EDGES,
    ids: { dataset: "dataset", checkpoint: "base_checkpoint", training: "mos" },
  },
};
