import type { WorkflowGraph, WorkflowNode, WorkflowSchema } from "./types";

const layout_x = 64;
const layout_y = 80;
const layout_column_gap = 64;
const layout_node_min_width = 220;
const layout_row_gap = 52;
const layout_panel_gap = 120;
const layout_panel_width = 280;
const layout_panel_gap_x = 32;

const layout_ordering_sweeps = 8;
const layout_coord_sweeps = 10;

const layout_dummy_height = 24;

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
  for (let sweep = 0; sweep < layout_ordering_sweeps; sweep += 1) {
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

  const heightOf = (id: string): number => (isDummy.has(id) ? layout_dummy_height : estimatedNodeHeight(schema, nodeById.get(id)!));
  const top = new Map<string, number>();
  for (const layer of layers) {
    let y = layout_y;
    for (const id of layer) {
      top.set(id, y);
      y += heightOf(id) + layout_row_gap;
    }
  }
  const centreOf = (id: string): number => (top.get(id) ?? layout_y) + heightOf(id) / 2;
  const desiredTopOf = (id: string): number => {
    const neighbours = [...(preds.get(id) ?? []), ...(succs.get(id) ?? [])].map(centreOf).sort((a, b) => a - b);
    return neighbours.length ? medianOf(neighbours) - heightOf(id) / 2 : top.get(id) ?? layout_y;
  };
  for (let sweep = 0; sweep < layout_coord_sweeps; sweep += 1) {
    for (let rank = 0; rank < layerCount; rank += 1) {
      const layer = layers[rank]!;
      if (sweep % 2 === 0) {
        let floor = layout_y;
        for (const id of layer) {
          const y = Math.max(desiredTopOf(id), floor);
          top.set(id, y);
          floor = y + heightOf(id) + layout_row_gap;
        }
      } else {
        let ceiling = Number.POSITIVE_INFINITY;
        for (let i = layer.length - 1; i >= 0; i -= 1) {
          const id = layer[i]!;
          const height = heightOf(id);
          const y = Math.min(desiredTopOf(id), ceiling - height);
          top.set(id, y);
          ceiling = y - layout_row_gap;
        }
      }
    }
  }

  const minTop = Math.min(layout_y, ...[...top.values()]);
  if (minTop < layout_y) {
    for (const [id, y] of top) top.set(id, y + (layout_y - minTop));
  }

  const layerX = [layout_x];
  for (let rank = 0; rank + 1 < layerCount; rank += 1) {
    const widths = layers[rank]!
      .filter((id) => !isDummy.has(id))
      .map((id) => {
        const width = nodeWidths.get(id);
        if (width === undefined) throw new Error(`Missing rendered width for node: ${id}`);
        return width;
      });
    const layerWidth = Math.max(layout_node_min_width, ...widths);
    layerX.push(layerX[rank]! + layerWidth + layout_column_gap);
  }

  let maxBottom = layout_y;
  const positioned = new Map<string, WorkflowNode>();
  for (let rank = 0; rank < layerCount; rank += 1) {
    for (const id of layers[rank]!) {
      if (isDummy.has(id)) continue; // routing dummies reserve space but are not rendered
      const node = nodeById.get(id)!;
      const y = top.get(id) ?? layout_y;
      positioned.set(id, { ...node, x: layerX[rank]!, y });
      maxBottom = Math.max(maxBottom, y + heightOf(id));
    }
  }

  const panels = (graph.panels ?? []).map((panel, index) => ({
    ...panel,
    x: layout_x + index * (layout_panel_width + layout_panel_gap_x),
    y: maxBottom + layout_panel_gap,
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
