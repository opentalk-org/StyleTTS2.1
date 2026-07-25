import type { SchemaValues } from "@/shared/schema-form/types";
import type { Viewport, WorkflowDefinition, WorkflowEdge, WorkflowGraph, WorkflowNode, WorkflowPayload, WorkflowRunContext, WorkflowSchema } from "./types";

const LAYOUT_X = 64;
const LAYOUT_Y = 80;
const LAYOUT_COLUMN_GAP = 64;
const LAYOUT_NODE_MIN_WIDTH = 220;
const LAYOUT_ROW_GAP = 52;
const LAYOUT_PANEL_GAP = 120;
const LAYOUT_PANEL_WIDTH = 280;
const LAYOUT_PANEL_GAP_X = 32;

export function typeAccepts(schema: WorkflowSchema, targetType: string, sourceType: string): boolean {
  if (!schema.types[targetType] || !schema.types[sourceType]) {
    throw new Error(`Unknown port type: ${targetType} or ${sourceType}`);
  }
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

export function deleteNodes(graph: WorkflowGraph, nodeIds: string[]): WorkflowGraph {
  const removed = new Set(nodeIds);
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !removed.has(node.id)),
    edges: graph.edges.filter((edge) => !removed.has(edge.source_node) && !removed.has(edge.target_node)),
    panels: (graph.panels ?? [])
      .map((panel) => ({
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
      source_node: edge.source_node === previous ? nextId : edge.source_node,
      source_port: edge.source_port,
      target_node: edge.target_node === previous ? nextId : edge.target_node,
      target_port: edge.target_port,
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

const LAYOUT_ORDERING_SWEEPS = 8;
const LAYOUT_COORD_SWEEPS = 10;

const LAYOUT_DUMMY_HEIGHT = 24;

export function autoLayoutGraph(schema: WorkflowSchema, graph: WorkflowGraph, nodeWidths: ReadonlyMap<string, number>): WorkflowGraph {
  if (graph.nodes.length === 0) return graph;

  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const order = topologicalNodeOrder(graph);
  const orderIndex = new Map(order.map((nodeId, index) => [nodeId, index]));

  const edges = graph.edges.filter(
    (edge) =>
      nodeById.has(edge.source_node) &&
      nodeById.has(edge.target_node) &&
      edge.source_node !== edge.target_node &&
      (orderIndex.get(edge.source_node) ?? 0) < (orderIndex.get(edge.target_node) ?? 0),
  );

  const layerOf = new Map(graph.nodes.map((node) => [node.id, 0]));
  for (const nodeId of order) {
    const base = layerOf.get(nodeId) ?? 0;
    for (const edge of edges) {
      if (edge.source_node !== nodeId) continue;
      layerOf.set(edge.target_node, Math.max(layerOf.get(edge.target_node) ?? 0, base + 1));
    }
  }
  const layerCount = Math.max(0, ...[...layerOf.values()]) + 1;

  const layers: string[][] = Array.from({ length: layerCount }, () => []);
  const itemLayer = new Map<string, number>();
  const isDummy = new Set<string>();
  for (const nodeId of order) {
    const layer = layerOf.get(nodeId) ?? 0;
    layers[layer]!.push(nodeId);
    itemLayer.set(nodeId, layer);
  }
  const succs = new Map<string, string[]>();
  const preds = new Map<string, string[]>();
  const linkItems = (source: string, target: string): void => {
    (succs.get(source) ?? succs.set(source, []).get(source)!).push(target);
    (preds.get(target) ?? preds.set(target, []).get(target)!).push(source);
  };
  let dummyCount = 0;
  for (const edge of edges) {
    const start = layerOf.get(edge.source_node) ?? 0;
    const finish = layerOf.get(edge.target_node) ?? 0;
    let previous = edge.source_node;
    for (let layer = start + 1; layer < finish; layer += 1) {
      const dummy = `__layout_dummy_${dummyCount++}`;
      isDummy.add(dummy);
      itemLayer.set(dummy, layer);
      layers[layer]!.push(dummy);
      linkItems(previous, dummy);
      previous = dummy;
    }
    linkItems(previous, edge.target_node);
  }
  const segments: Array<[string, string]> = [];
  for (const [source, targets] of succs) for (const target of targets) segments.push([source, target]);

  const positionInLayer = (): Map<string, number> => {
    const pos = new Map<string, number>();
    for (const layer of layers) layer.forEach((id, index) => pos.set(id, index));
    return pos;
  };

  const crossings = (): number => {
    const pos = positionInLayer();
    let total = 0;
    for (let rank = 0; rank + 1 < layerCount; rank += 1) {
      const spans = segments
        .filter(([source]) => itemLayer.get(source) === rank)
        .map(([source, target]) => [pos.get(source) ?? 0, pos.get(target) ?? 0] as const)
        .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
      for (let i = 0; i < spans.length; i += 1) {
        for (let j = i + 1; j < spans.length; j += 1) {
          if (spans[i]![1] > spans[j]![1]) total += 1;
        }
      }
    }
    return total;
  };

  let bestLayers = layers.map((layer) => [...layer]);
  let bestCrossings = crossings();
  for (let sweep = 0; sweep < LAYOUT_ORDERING_SWEEPS; sweep += 1) {
    const downward = sweep % 2 === 0;
    const neighboursOf = downward ? preds : succs;
    const from = downward ? 1 : layerCount - 2;
    const to = downward ? layerCount : -1;
    const step = downward ? 1 : -1;
    const pos = positionInLayer();
    for (let rank = from; rank !== to; rank += step) {
      const ranked = layers[rank]!.map((id, index) => {
        const neighbours = (neighboursOf.get(id) ?? []).map((n) => pos.get(n) ?? 0).sort((a, b) => a - b);
        return { id, index, key: neighbours.length ? medianOf(neighbours) : index };
      });
      ranked.sort((left, right) => left.key - right.key || left.index - right.index);
      layers[rank] = ranked.map((entry) => entry.id);
      layers[rank]!.forEach((id, index) => pos.set(id, index));
    }
    const current = crossings();
    if (current < bestCrossings) {
      bestCrossings = current;
      bestLayers = layers.map((layer) => [...layer]);
    }
  }
  for (let rank = 0; rank < layerCount; rank += 1) layers[rank] = bestLayers[rank]!;

  const heightOf = (id: string): number => (isDummy.has(id) ? LAYOUT_DUMMY_HEIGHT : estimatedNodeHeight(schema, nodeById.get(id)!));
  const top = new Map<string, number>();
  for (const layer of layers) {
    let y = LAYOUT_Y;
    for (const id of layer) {
      top.set(id, y);
      y += heightOf(id) + LAYOUT_ROW_GAP;
    }
  }
  const centreOf = (id: string): number => (top.get(id) ?? LAYOUT_Y) + heightOf(id) / 2;
  const desiredTopOf = (id: string): number => {
    const neighbours = [...(preds.get(id) ?? []), ...(succs.get(id) ?? [])].map(centreOf).sort((a, b) => a - b);
    return neighbours.length ? medianOf(neighbours) - heightOf(id) / 2 : top.get(id) ?? LAYOUT_Y;
  };
  for (let sweep = 0; sweep < LAYOUT_COORD_SWEEPS; sweep += 1) {
    for (let rank = 0; rank < layerCount; rank += 1) {
      const layer = layers[rank]!;
      if (sweep % 2 === 0) {
        let floor = LAYOUT_Y;
        for (const id of layer) {
          const y = Math.max(desiredTopOf(id), floor);
          top.set(id, y);
          floor = y + heightOf(id) + LAYOUT_ROW_GAP;
        }
      } else {
        let ceiling = Number.POSITIVE_INFINITY;
        for (let i = layer.length - 1; i >= 0; i -= 1) {
          const id = layer[i]!;
          const height = heightOf(id);
          const y = Math.min(desiredTopOf(id), ceiling - height);
          top.set(id, y);
          ceiling = y - LAYOUT_ROW_GAP;
        }
      }
    }
  }

  const minTop = Math.min(LAYOUT_Y, ...[...top.values()]);
  if (minTop < LAYOUT_Y) {
    for (const [id, y] of top) top.set(id, y + (LAYOUT_Y - minTop));
  }

  const layerX = [LAYOUT_X];
  for (let rank = 0; rank + 1 < layerCount; rank += 1) {
    const widths = layers[rank]!
      .filter((id) => !isDummy.has(id))
      .map((id) => {
        const width = nodeWidths.get(id);
        if (width === undefined) throw new Error(`Missing rendered width for node: ${id}`);
        return width;
      });
    const layerWidth = Math.max(LAYOUT_NODE_MIN_WIDTH, ...widths);
    layerX.push(layerX[rank]! + layerWidth + LAYOUT_COLUMN_GAP);
  }

  let maxBottom = LAYOUT_Y;
  const positioned = new Map<string, WorkflowNode>();
  for (let rank = 0; rank < layerCount; rank += 1) {
    for (const id of layers[rank]!) {
      if (isDummy.has(id)) continue; // routing dummies reserve space but are not rendered
      const node = nodeById.get(id)!;
      const y = top.get(id) ?? LAYOUT_Y;
      positioned.set(id, { ...node, x: layerX[rank]!, y });
      maxBottom = Math.max(maxBottom, y + heightOf(id));
    }
  }

  const panels = (graph.panels ?? []).map((panel, index) => ({
    ...panel,
    x: LAYOUT_X + index * (LAYOUT_PANEL_WIDTH + LAYOUT_PANEL_GAP_X),
    y: maxBottom + LAYOUT_PANEL_GAP,
  }));

  return {
    ...graph,
    nodes: graph.nodes.map((node) => positioned.get(node.id) ?? node),
    panels,
  };
}

function medianOf(sorted: number[]): number {
  const count = sorted.length;
  if (count === 0) return 0;
  const mid = Math.floor(count / 2);
  return count % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
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
  return { nodes: payload.nodes, edges: payload.edges, panels: graph.panels ?? [], context: payload.context, launch_source: null };
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
    const resources = nodeResourceRequirements(schema, node);
    for (const [key, amount] of Object.entries(resources)) {
      requirements[key] = Math.max(requirements[key] ?? 0, amount);
    }
  }
  return requirements;
}

export function nodeResourceRequirements(schema: WorkflowSchema, node: WorkflowNode): Record<string, number> {
  const info = schema.nodes[node.type];
  if (!info) throw new Error(`Unknown node type: ${node.type}`);
  const runtime = mergeRuntimeDefaults(info.runtime_defaults, node.runtime);
  const policy = recordValue(runtime["resource_policy"], `${node.id} resource policy`);
  return numberRecord(policy["resources"], `${node.id} resource requirements`);
}

function mergeRuntimeDefaults(defaults: SchemaValues, runtime: SchemaValues): SchemaValues {
  const merged = { ...defaults };
  for (const [key, value] of Object.entries(runtime)) {
    const current = merged[key];
    if (isRecord(current) && isRecord(value)) merged[key] = { ...current, ...value };
    else merged[key] = value;
  }
  return merged;
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function topologicalNodeOrder(graph: WorkflowGraph): string[] {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const original = [...graph.nodes].sort((left, right) => left.x - right.x || left.y - right.y || left.id.localeCompare(right.id));
  const indegree = new Map(graph.nodes.map((node) => [node.id, 0]));
  const outgoing = new Map(graph.nodes.map((node) => [node.id, [] as string[]]));
  for (const edge of graph.edges) {
    if (!nodes.has(edge.source_node) || !nodes.has(edge.target_node)) continue;
    indegree.set(edge.target_node, (indegree.get(edge.target_node) ?? 0) + 1);
    outgoing.get(edge.source_node)?.push(edge.target_node);
  }

  const ready = original.filter((node) => (indegree.get(node.id) ?? 0) === 0).map((node) => node.id);
  const out: string[] = [];
  while (ready.length > 0) {
    ready.sort((left, right) => {
      const leftNode = nodes.get(left);
      const rightNode = nodes.get(right);
      if (!leftNode || !rightNode) return left.localeCompare(right);
      return leftNode.x - rightNode.x || leftNode.y - rightNode.y || left.localeCompare(right);
    });
    const nodeId = ready.shift()!;
    out.push(nodeId);
    for (const target of outgoing.get(nodeId) ?? []) {
      const next = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, next);
      if (next === 0) ready.push(target);
    }
  }

  const seen = new Set(out);
  for (const node of original) {
    if (!seen.has(node.id)) out.push(node.id);
  }
  return out;
}

function estimatedNodeHeight(schema: WorkflowSchema, node: WorkflowNode): number {
  const info = schema.nodes[node.type];
  if (!info) return 170;
  const portRows = Math.max(Object.keys(info.inputs).length, Object.keys(info.outputs).length, 1);
  return 122 + portRows * 28;
}
