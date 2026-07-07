import type { Option } from "@/shared/ui/Select";

import type { WorkflowEdge } from "../workflows/types";

export type TestingMode = "single" | "sweep";

export type TestingNodeIds = {
  run: string;
  prompt: string;
  checkpoint: string;
  dataset: string;
  styleRef?: string;
  synthesis?: string;
  saveAudio?: string;
  styleSweep?: string;
  sweepSynthesis?: string;
};

export type TestingWorkflowSpec = {
  value: TestingMode;
  label: string;
  nodes: { id: string; type: string; x: number; y: number }[];
  edges: WorkflowEdge[];
  ids: TestingNodeIds;
};

const SINGLE_NODES = [
  { id: "run", type: "TestingRunInput", x: -160, y: 380 },
  { id: "prompt", type: "TestingTextPrompt", x: 64, y: 220 },
  { id: "checkpoint", type: "SelectCheckpoint", x: 64, y: 420 },
  { id: "prefetch_checkpoint", type: "PrefetchCheckpoint", x: 330, y: 420 },
  { id: "style_ref", type: "SelectStyleReference", x: 64, y: 620 },
  { id: "synthesis", type: "StyleTtsSynthesis", x: 640, y: 400 },
  { id: "save_audio", type: "SaveAudioRecord", x: 910, y: 400 },
  { id: "to_dataset", type: "AddAudioToDataset", x: 1160, y: 400 },
];

const SINGLE_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "prompt", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "checkpoint", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "style_ref", target_port: "run" },
  { source_node: "checkpoint", source_port: "checkpoint_ref", target_node: "prefetch_checkpoint", target_port: "checkpoint_ref" },
  { source_node: "prefetch_checkpoint", source_port: "checkpoint", target_node: "synthesis", target_port: "checkpoint" },
  { source_node: "prompt", source_port: "prompt_text", target_node: "synthesis", target_port: "prompt_text" },
  { source_node: "style_ref", source_port: "style_reference", target_node: "synthesis", target_port: "style_reference" },
  { source_node: "synthesis", source_port: "audio", target_node: "save_audio", target_port: "audio" },
  { source_node: "save_audio", source_port: "audio", target_node: "to_dataset", target_port: "audio" },
];

const SWEEP_NODES = [
  { id: "run", type: "TestingRunInput", x: -160, y: 320 },
  { id: "prompt", type: "TestingTextPrompt", x: 64, y: 180 },
  { id: "checkpoint", type: "SelectCheckpoint", x: 64, y: 380 },
  { id: "prefetch_checkpoint", type: "PrefetchCheckpoint", x: 330, y: 380 },
  { id: "style_sweep", type: "StyleReferenceSweep", x: 64, y: 560 },
  { id: "sweep_synthesis", type: "StyleTtsSynthesis", x: 640, y: 360 },
  { id: "save_audio", type: "SaveAudioRecord", x: 910, y: 360 },
  { id: "to_dataset", type: "AddAudioToDataset", x: 1160, y: 360 },
];

const SWEEP_EDGES: WorkflowEdge[] = [
  { source_node: "run", source_port: "run", target_node: "prompt", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "checkpoint", target_port: "run" },
  { source_node: "run", source_port: "run", target_node: "style_sweep", target_port: "run" },
  { source_node: "checkpoint", source_port: "checkpoint_ref", target_node: "prefetch_checkpoint", target_port: "checkpoint_ref" },
  { source_node: "prefetch_checkpoint", source_port: "checkpoint", target_node: "sweep_synthesis", target_port: "checkpoint" },
  { source_node: "prompt", source_port: "prompt_text", target_node: "sweep_synthesis", target_port: "prompt_text" },
  { source_node: "style_sweep", source_port: "style_reference", target_node: "sweep_synthesis", target_port: "style_reference" },
  { source_node: "sweep_synthesis", source_port: "audio", target_node: "save_audio", target_port: "audio" },
  { source_node: "save_audio", source_port: "audio", target_node: "to_dataset", target_port: "audio" },
];

export const TESTING_WORKFLOWS: Record<TestingMode, TestingWorkflowSpec> = {
  single: {
    value: "single",
    label: "Single synthesis",
    nodes: SINGLE_NODES,
    edges: SINGLE_EDGES,
    ids: {
      run: "run",
      prompt: "prompt",
      checkpoint: "checkpoint",
      dataset: "to_dataset",
      styleRef: "style_ref",
      synthesis: "synthesis",
      saveAudio: "save_audio",
    },
  },
  sweep: {
    value: "sweep",
    label: "Voice sweep",
    nodes: SWEEP_NODES,
    edges: SWEEP_EDGES,
    ids: {
      run: "run",
      prompt: "prompt",
      checkpoint: "checkpoint",
      dataset: "to_dataset",
      styleSweep: "style_sweep",
      sweepSynthesis: "sweep_synthesis",
      saveAudio: "save_audio",
    },
  },
};

export const TESTING_OPTIONS: Option[] = Object.values(TESTING_WORKFLOWS).map((workflow) => ({
  value: workflow.value,
  label: workflow.label,
}));
