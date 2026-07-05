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

export function graphPayload(graph: WorkflowGraph, runId: string | null, context: WorkflowRunContext): WorkflowPayload {
  return {
    run_id: runId,
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
