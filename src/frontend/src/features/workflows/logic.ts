import type { SchemaValues } from "@/shared/schema-form/types";
import type { Viewport, WorkflowDefinition, WorkflowEdge, WorkflowGraph, WorkflowNode, WorkflowPayload, WorkflowRunContext, WorkflowSchema } from "./types";

export function typeAccepts(schema: WorkflowSchema, targetType: string, sourceType: string): boolean {
  const target = schema.types[targetType];
  const source = schema.types[sourceType];
  if (!target || !source) throw new Error(`Unknown port type: ${targetType} or ${sourceType}`);
  if (target.members.length) {
    if (source.members.length) return source.members.every((member) => target.members.includes(member));
    return target.members.includes(sourceType);
  }
  if (source.members.length) return source.members.every((member) => typeAccepts(schema, targetType, member));
  return targetType === sourceType;
}

export function nodeAccent(schema: WorkflowSchema, nodeType: string): string {
  const info = schema.nodes[nodeType];
  if (!info) throw new Error(`Unknown node type: ${nodeType}`);
  const outputs = Object.values(info.outputs);
  if (!outputs.length) return "#3a4353";
  const output = outputs[0];
  if (!output) throw new Error(`Node has no output: ${nodeType}`);
  const type = schema.types[output.type];
  if (!type) throw new Error(`Unknown output type: ${output.type}`);
  return type.color;
}

export function addNode(schema: WorkflowSchema, graph: WorkflowGraph, type: string, x: number, y: number): WorkflowGraph {
  const count = graph.nodes.filter((node) => node.type === type).length + 1;
  const info = schema.nodes[type];
  if (!info) throw new Error(`Unknown node type: ${type}`);
  const node: WorkflowNode = {
    id: `${type}_${count}`,
    type,
    x,
    y,
    params: structuredClone(info.settings_defaults) as SchemaValues,
    runtime: structuredClone(info.runtime_defaults) as SchemaValues,
  };
  return { ...graph, nodes: [...graph.nodes, node] };
}

export function deleteNode(graph: WorkflowGraph, nodeId: string): WorkflowGraph {
  return {
    nodes: graph.nodes.filter((node) => node.id !== nodeId),
    edges: graph.edges.filter((edge) => edge.source_node !== nodeId && edge.target_node !== nodeId),
  };
}

export function deleteNodes(graph: WorkflowGraph, nodeIds: string[]): WorkflowGraph {
  const removed = new Set(nodeIds);
  return {
    nodes: graph.nodes.filter((node) => !removed.has(node.id)),
    edges: graph.edges.filter((edge) => !removed.has(edge.source_node) && !removed.has(edge.target_node)),
  };
}

export function renameNode(graph: WorkflowGraph, previous: string, nextId: string): WorkflowGraph {
  return {
    nodes: graph.nodes.map((node) => (node.id === previous ? { ...node, id: nextId } : node)),
    edges: graph.edges.map((edge) => ({
      source_node: edge.source_node === previous ? nextId : edge.source_node,
      source_port: edge.source_port,
      target_node: edge.target_node === previous ? nextId : edge.target_node,
      target_port: edge.target_port,
    })),
  };
}

export function connect(schema: WorkflowSchema, graph: WorkflowGraph, edge: WorkflowEdge): WorkflowGraph {
  const source = graph.nodes.find((node) => node.id === edge.source_node);
  const target = graph.nodes.find((node) => node.id === edge.target_node);
  if (!source || !target || source.id === target.id) return graph;
  const sourceInfo = schema.nodes[source.type];
  const targetInfo = schema.nodes[target.type];
  if (!sourceInfo || !targetInfo) throw new Error("Graph contains unknown node type");
  const sourcePort = sourceInfo.outputs[edge.source_port];
  const targetPort = targetInfo.inputs[edge.target_port];
  if (!sourcePort || !targetPort) throw new Error("Graph contains unknown port");
  const sourceType = sourcePort.type;
  const targetType = targetPort.type;
  if (!typeAccepts(schema, targetType, sourceType)) return graph;
  const duplicate = graph.edges.some((item) => JSON.stringify(item) === JSON.stringify(edge));
  if (duplicate) return graph;
  return { ...graph, edges: [...graph.edges, edge] };
}

export function moveNodes(graph: WorkflowGraph, nodeIds: string[], dx: number, dy: number): WorkflowGraph {
  const selected = new Set(nodeIds);
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (selected.has(node.id) ? { ...node, x: node.x + dx, y: node.y + dy } : node)),
  };
}

export function graphPoint(viewport: Viewport, clientX: number, clientY: number, left: number, top: number) {
  return {
    x: (clientX - left - viewport.x) / viewport.zoom,
    y: (clientY - top - viewport.y) / viewport.zoom,
  };
}

export function zoomViewport(viewport: Viewport, nextZoom: number, anchorX: number, anchorY: number): Viewport {
  const zoom = Math.max(0.25, Math.min(2, nextZoom));
  const graphX = (anchorX - viewport.x) / viewport.zoom;
  const graphY = (anchorY - viewport.y) / viewport.zoom;
  return { x: anchorX - graphX * zoom, y: anchorY - graphY * zoom, zoom };
}

export function graphPayload(graph: WorkflowGraph, runId: string | null, context: WorkflowRunContext, runnerId: string | null = null): WorkflowPayload {
  return {
    run_id: runId,
    runner_id: runnerId,
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      x: node.x,
      y: node.y,
      params: node.params,
      runtime: node.runtime,
    })),
    edges: graph.edges,
    context,
  };
}

export function defaultWorkflowContext(config: SchemaValues): WorkflowRunContext {
  return { work_dir: "work", cache_dir: "cache", output_dir: "outputs", device: "cuda", config, input_items: [] };
}

export function workflowDefinition(graph: WorkflowGraph, config: SchemaValues): WorkflowDefinition {
  const payload = graphPayload(graph, null, defaultWorkflowContext(config));
  return { nodes: payload.nodes, edges: payload.edges, context: payload.context, launch_source: null };
}

export function runtimeConfigForGraph(schema: WorkflowSchema, graph: WorkflowGraph, config: SchemaValues): SchemaValues {
  return { ...config, resources: resourceLimitsForGraph(schema, graph, numberRecord(config["resources"], "runtime resources")) };
}

export function resourceLimitsForGraph(schema: WorkflowSchema, graph: WorkflowGraph, currentLimits: Record<string, number>): Record<string, number> {
  const requirements = resourceRequirementsForGraph(schema, graph);
  return Object.fromEntries(
    Object.entries(requirements)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, amount]) => [key, currentLimits[key] ?? amount]),
  );
}

export function resourceRequirementsForGraph(schema: WorkflowSchema, graph: WorkflowGraph): Record<string, number> {
  const requirements: Record<string, number> = {};
  for (const node of graph.nodes) {
    const info = schema.nodes[node.type];
    if (!info) throw new Error(`Unknown node type: ${node.type}`);
    const resources = nodeResourceRequirements(node);
    for (const [key, amount] of Object.entries(resources)) {
      requirements[key] = Math.max(requirements[key] ?? 0, amount);
    }
  }
  return requirements;
}

export function nodeResourceRequirements(node: WorkflowNode): Record<string, number> {
  const policy = recordValue(node.runtime["resource_policy"], `${node.id} resource policy`);
  return numberRecord(policy["resources"], `${node.id} resource requirements`);
}

export function numberRecord(value: unknown, label: string): Record<string, number> {
  const record = recordValue(value, label);
  for (const [key, item] of Object.entries(record)) {
    if (typeof item !== "number" || Number.isNaN(item)) throw new Error(`${label} contains a non-number value: ${key}`);
  }
  return record as Record<string, number>;
}

function recordValue(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
}
