import type { JsonSchema, SchemaValues } from "@/shared/schema-form/types";
import type { Option } from "@/shared/ui/Select";

import type { Checkpoint } from "../checkpoints/api";
import type { Voice } from "../voices/api";
import { typeAccepts } from "../workflows/logic";
import type { WorkflowGraph, WorkflowNode, WorkflowSchema } from "../workflows/types";
import type { TestingWorkflowSpec } from "./workflows";

export { TESTING_OPTIONS, TESTING_WORKFLOWS, type TestingMode, type TestingWorkflowSpec } from "./workflows";

export type SingleConfig = {
  ckpt: string;
  dataset: string;
  text: string;
  lang: string;
  steps: number;
  emb: number;
  styleRef: string;
  alpha: number;
  beta: number;
};

export type SweepConfig = {
  ckpt: string;
  dataset: string;
  text: string;
  voices: { id: string; name: string }[];
  n: number;
  alpha: number;
  beta: number;
};

export function createTestingGraph(schema: WorkflowSchema, spec: TestingWorkflowSpec): WorkflowGraph {
  assertTestingNodes(schema, spec);
  assertTestingEdges(schema, spec);
  return {
    nodes: spec.nodes.map((node) => {
      const info = schema.nodes[node.type];
      if (!info) throw new Error(`Testing node is not registered: ${node.type}`);
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

export function singleConfigFromGraph(graph: WorkflowGraph, spec: TestingWorkflowSpec): SingleConfig {
  const prompt = testingNode(graph, spec.ids.prompt);
  const checkpoint = testingNode(graph, spec.ids.checkpoint);
  const styleRef = testingNode(graph, requiredNodeId(spec.ids.styleRef, "single style reference"));
  const synthesis = testingNode(graph, requiredNodeId(spec.ids.synthesis, "single synthesis"));
  const dataset = testingNode(graph, spec.ids.dataset);
  return {
    ckpt: String(checkpoint.params.checkpoint_id),
    dataset: String(dataset.params.dataset_id ?? ""),
    text: String(prompt.params.text),
    lang: String(prompt.params.language),
    steps: Number(synthesis.params.diffusion_steps),
    emb: Number(synthesis.params.embedding_scale),
    styleRef: String(styleRef.params.reference_id),
    alpha: Number(styleRef.params.alpha),
    beta: Number(styleRef.params.beta),
  };
}

export function sweepConfigFromGraph(graph: WorkflowGraph, spec: TestingWorkflowSpec, availableVoices: Voice[] = []): SweepConfig {
  const prompt = testingNode(graph, spec.ids.prompt);
  const checkpoint = testingNode(graph, spec.ids.checkpoint);
  const styleSweep = testingNode(graph, requiredNodeId(spec.ids.styleSweep, "sweep style references"));
  const voiceIds = stringArrayParam(styleSweep.params.voices);
  const dataset = testingNode(graph, spec.ids.dataset);
  return {
    ckpt: String(checkpoint.params.checkpoint_id),
    dataset: String(dataset.params.dataset_id ?? ""),
    text: String(prompt.params.text),
    voices: voiceIds.map((id) => ({ id, name: voiceName(id, availableVoices) })),
    n: Number(styleSweep.params.samples_per_voice),
    alpha: Number(styleSweep.params.alpha),
    beta: Number(styleSweep.params.beta),
  };
}

export function testingNode(graph: WorkflowGraph, nodeId: string): WorkflowNode {
  const node = graph.nodes.find((item) => item.id === nodeId);
  if (!node) throw new Error(`Testing graph is missing node: ${nodeId}`);
  return node;
}

export function updateNodeParams(graph: WorkflowGraph, nodeId: string, params: SchemaValues): WorkflowGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === nodeId ? { ...node, params } : node)),
  };
}

export function numericSetting(schema: WorkflowSchema, node: WorkflowNode, name: string): { min: number; max: number } {
  const prop = settingProp(schema, node, name);
  if (typeof prop.minimum !== "number" || typeof prop.maximum !== "number") {
    throw new Error(`Testing numeric setting is missing bounds: ${node.type}.${name}`);
  }
  return { min: prop.minimum, max: prop.maximum };
}

export function checkpointOptions(checkpoints: Checkpoint[]): Option[] {
  const rows = checkpoints.filter((checkpoint) => checkpoint.type_ === "styletts2");
  return [
    { value: "", label: rows.length ? "— select checkpoint —" : "No StyleTTS2 checkpoints available" },
    ...rows.map((checkpoint) => ({ value: checkpoint.id, label: checkpoint.name })),
  ];
}

export function assertTestingNodes(schema: WorkflowSchema, spec: TestingWorkflowSpec) {
  const missing = spec.nodes.map((node) => node.type).filter((type) => !schema.nodes[type]);
  if (missing.length > 0) {
    throw new Error(`Testing workflow nodes are not registered: ${missing.join(", ")}`);
  }
}

function assertTestingEdges(schema: WorkflowSchema, spec: TestingWorkflowSpec) {
  const nodes = new Map(spec.nodes.map((node) => [node.id, node.type]));
  for (const edge of spec.edges) {
    const sourceType = nodes.get(edge.source_node);
    const targetType = nodes.get(edge.target_node);
    if (!sourceType || !targetType) throw new Error(`Testing workflow edge references unknown node: ${JSON.stringify(edge)}`);
    const source = schema.nodes[sourceType];
    const target = schema.nodes[targetType];
    if (!source || !target) throw new Error(`Testing workflow edge references unregistered node type: ${JSON.stringify(edge)}`);
    const sourcePort = source.outputs[edge.source_port];
    const targetPort = target.inputs[edge.target_port];
    if (!sourcePort || !targetPort) throw new Error(`Testing workflow edge references unknown port: ${JSON.stringify(edge)}`);
    if (!typeAccepts(schema, targetPort.type, sourcePort.type)) throw new Error(`Testing workflow edge has incompatible types: ${JSON.stringify(edge)}`);
  }
}

function settingProp(schema: WorkflowSchema, node: WorkflowNode, name: string): JsonSchema {
  const info = schema.nodes[node.type];
  if (!info) throw new Error(`Testing node is not registered: ${node.type}`);
  const prop = info.settings.properties?.[name];
  if (!prop) throw new Error(`Testing setting is not declared by node schema: ${node.type}.${name}`);
  return prop;
}

function requiredNodeId(nodeId: string | undefined, label: string): string {
  if (!nodeId) throw new Error(`Testing workflow is missing ${label} node id`);
  return nodeId;
}

function stringArrayParam(value: unknown): string[] {
  if (!Array.isArray(value)) throw new Error("Testing sweep voices must be an array");
  return value.map((item) => String(item));
}

function voiceName(id: string, voices: Voice[]): string {
  const voice = voices.find((item) => item.id === id);
  return voice ? voice.name : id;
}
