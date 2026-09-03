import type { SchemaValues } from "@/shared/schema-form/types";
import type { WorkflowEdge, WorkflowGraph, WorkflowNode, WorkflowSchema } from "./types";

export function typeAccepts(schema: WorkflowSchema, targetType: string, sourceType: string): boolean {
  if (!schema.types[targetType] || !schema.types[sourceType]) {
    throw new Error(`Unknown port type: ${targetType} or ${sourceType}`);
  }
  return targetType === sourceType;
}

export function nodeAccent(schema: WorkflowSchema, nodeType: string): string {
  const info = schema.nodes[nodeType];
  if (!info) throw new Error(`Unknown node type: ${nodeType}`);
  const output = Object.values(info.outputs)[0];
  if (!output) return "#3a4353";
  const type = schema.types[output.type];
  if (!type) throw new Error(`Unknown output type: ${output.type}`);
  return type.color;
}

export function addNode(schema: WorkflowSchema, graph: WorkflowGraph, type: string, x: number, y: number): WorkflowGraph {
  const info = schema.nodes[type];
  if (!info) throw new Error(`Unknown node type: ${type}`);
  const count = graph.nodes.filter((node) => node.type === type).length + 1;
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

export function deleteNodes(graph: WorkflowGraph, nodeIds: string[]): WorkflowGraph {
  const removed = new Set(nodeIds);
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !removed.has(node.id)),
    edges: graph.edges.filter((edge) => !removed.has(edge.source_node) && !removed.has(edge.target_node)),
    panels: (graph.panels ?? []).map((panel) => ({
      ...panel,
      controls: panel.controls
        .map((control) => ({ ...control, targets: control.targets.filter((target) => !removed.has(target.node_id)) }))
        .filter((control) => control.targets.length > 0),
    })),
  };
}

export function renameNode(graph: WorkflowGraph, previous: string, nextId: string): WorkflowGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === previous ? { ...node, id: nextId } : node)),
    edges: graph.edges.map((edge) => ({
      ...edge,
      source_node: edge.source_node === previous ? nextId : edge.source_node,
      target_node: edge.target_node === previous ? nextId : edge.target_node,
    })),
    panels: (graph.panels ?? []).map((panel) => ({
      ...panel,
      controls: panel.controls.map((control) => ({
        ...control,
        targets: control.targets.map((target) => (target.node_id === previous ? { ...target, node_id: nextId } : target)),
      })),
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
  if (!typeAccepts(schema, targetPort.type, sourcePort.type)) return graph;
  const duplicate = graph.edges.some((item) =>
    item.source_node === edge.source_node && item.source_port === edge.source_port
    && item.target_node === edge.target_node && item.target_port === edge.target_port,
  );
  return duplicate ? graph : { ...graph, edges: [...graph.edges, edge] };
}

export function moveNodes(graph: WorkflowGraph, nodeIds: string[], dx: number, dy: number): WorkflowGraph {
  const selected = new Set(nodeIds);
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (selected.has(node.id) ? { ...node, x: node.x + dx, y: node.y + dy } : node)),
  };
}
