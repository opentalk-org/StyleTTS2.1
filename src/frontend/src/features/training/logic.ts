import type { SchemaValues } from "@/shared/schema-form/types";
import type { Option } from "@/shared/ui/Select";

import type { FileAsset } from "../assets/api";
import type { Checkpoint } from "../checkpoints/api";
import { checkpointDecoderType, checkpointMultispeaker, checkpointSymbolCount as checkpointSymbolCountFromMetadata } from "../checkpoints/logic";
import type { Dataset } from "../datasets/api";
import { typeAccepts } from "../workflows/logic";
import type { WorkflowEdge, WorkflowGraph, WorkflowNode, WorkflowSchema } from "../workflows/types";
import type { TrainTab } from "./store";
import { TRAINING_WORKFLOWS } from "./workflowSpecs";

export { TRAINING_WORKFLOWS };

export type OodSetValue = { id: string; name: string; line_count: number };
export type TrainingNodeIds = {
  dataset: string;
  checkpoint: string;
  training: string;
  assets?: string;
  alphabet?: string;
};
export type TrainingSpecNode = {
  id: string;
  type: string;
  x: number;
  y: number;
  params?: Record<string, unknown>;
};
export type TrainingWorkflowSpec = {
  value: TrainTab;
  label: string;
  nodes: TrainingSpecNode[];
  edges: WorkflowEdge[];
  ids: TrainingNodeIds;
};

export const TRAINING_OPTIONS: Option[] = Object.values(TRAINING_WORKFLOWS).map((workflow) => ({
  value: workflow.value,
  label: workflow.label,
}));

export function createTrainingGraph(schema: WorkflowSchema, spec: TrainingWorkflowSpec): WorkflowGraph {
  assertTrainingNodes(schema, spec);
  assertTrainingEdges(schema, spec);
  return {
    nodes: spec.nodes.map((node) => {
      const info = schema.nodes[node.type];
      if (!info) throw new Error(`Training node is not registered: ${node.type}`);
      return {
        id: node.id,
        type: node.type,
        x: node.x,
        y: node.y,
        params: { ...structuredClone(info.settings_defaults), ...(node.params ?? {}) },
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

export function updateTrainingParams(graph: WorkflowGraph, spec: TrainingWorkflowSpec, params: SchemaValues): WorkflowGraph {
  return updateNodeParams(graph, spec.ids.training, params);
}

export function assertTrainingNodes(schema: WorkflowSchema, spec: TrainingWorkflowSpec) {
  const missing = spec.nodes.map((node) => node.type).filter((type) => !schema.nodes[type]);
  if (missing.length > 0) throw new Error(`Training workflow nodes are not registered: ${missing.join(", ")}`);
}

function assertTrainingEdges(schema: WorkflowSchema, spec: TrainingWorkflowSpec) {
  const nodes = new Map(spec.nodes.map((node) => [node.id, node.type]));
  for (const edge of spec.edges) {
    const sourceType = nodes.get(edge.source_node);
    const targetType = nodes.get(edge.target_node);
    if (!sourceType || !targetType) throw new Error(`Training workflow edge references unknown node: ${JSON.stringify(edge)}`);
    const source = schema.nodes[sourceType];
    const target = schema.nodes[targetType];
    if (!source || !target) throw new Error(`Training workflow edge references unregistered node type: ${JSON.stringify(edge)}`);
    const sourcePort = source.outputs[edge.source_port];
    const targetPort = target.inputs[edge.target_port];
    if (!sourcePort || !targetPort) throw new Error(`Training workflow edge references unknown port: ${JSON.stringify(edge)}`);
    if (!typeAccepts(schema, targetPort.type, sourcePort.type)) throw new Error(`Training workflow edge has incompatible types: ${JSON.stringify(edge)}`);
  }
}

export function datasetOptions(datasets: Dataset[]): Option[] {
  return [
    { value: "", label: datasets.length ? "— select training dataset —" : "No datasets available" },
    ...datasets.map((dataset) => ({ value: dataset.id, label: `${dataset.name} (${dataset.files.toLocaleString()} files)` })),
  ];
}

// Base-checkpoint dropdown sentinel that trains StyleTTS from random init. Kept
// in sync with SCRATCH_CHECKPOINT_ID in runner/nodes/assets/checkpoints.py.
export const SCRATCH_CHECKPOINT_ID = "__scratch__";

export function checkpointOptions(checkpoints: Checkpoint[], type: string, placeholder: string, includeScratch = false): Option[] {
  const rows = checkpoints.filter((checkpoint) => checkpointTypeKey(checkpoint.type_) === checkpointTypeKey(type));
  const scratch = includeScratch ? [{ value: SCRATCH_CHECKPOINT_ID, label: "From scratch (random init)" }] : [];
  return [
    { value: "", label: rows.length ? placeholder : `No ${type} checkpoints available` },
    ...scratch,
    ...rows.map((checkpoint) => ({ value: checkpoint.id, label: checkpoint.name })),
  ];
}

export function pretrainedAssetOptions(files: FileAsset[], checkpoints: Checkpoint[], type: string, placeholder: string): Option[] {
  const checkpointRows = checkpoints.filter((checkpoint) => checkpointTypeKey(checkpoint.type_) === checkpointTypeKey(type));
  const rows = [
    ...files.map((asset) => ({ value: asset.id, label: asset.name })),
    ...checkpointRows.map((checkpoint) => ({ value: checkpoint.id, label: `${checkpoint.name} · checkpoint` })),
  ];
  return [{ value: "", label: rows.length ? placeholder : "No assets available" }, ...rows];
}

export function oodSetValues(assets: FileAsset[]): OodSetValue[] {
  return assets.map((asset) => ({ id: asset.id, name: asset.name, line_count: lineCount(asset) }));
}

export function checkpointSymbolCount(checkpoint: Checkpoint | undefined): number | null {
  return checkpointSymbolCountFromMetadata(checkpoint);
}

export function styleTtsParamsForBaseCheckpoint(checkpoint: Checkpoint | undefined, current: SchemaValues): SchemaValues {
  const decoder = checkpointDecoderType(checkpoint);
  const multispeaker = checkpointMultispeaker(checkpoint);
  return {
    ...current,
    ...(decoder === "hifigan" || decoder === "istftnet" ? { decoder } : {}),
    ...(multispeaker === null ? {} : { multispeaker }),
  };
}

function lineCount(asset: FileAsset): number {
  const value = Number(asset.metadata.line_count);
  return Number.isFinite(value) ? value : 0;
}

function checkpointTypeKey(type: string): string {
  const normalized = type.trim().toLowerCase();
  const aliases: Record<string, string> = {
    styletts2: "styletts2", "styletts2_model": "styletts2", f0: "f0", f0_model: "f0",
    asr: "asr", asr_bundle: "asr", plbert: "plbert", "pl-bert": "plbert",
  };
  return aliases[normalized] ?? normalized;
}
