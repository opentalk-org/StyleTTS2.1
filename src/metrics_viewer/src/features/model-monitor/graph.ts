import { MarkerType, type Edge, type Node } from "@xyflow/react";

import { hueLine } from "./palette";
import type { LevelEdge, LevelNode, LevelView } from "./level";
import { layeredLayout, type LayoutBox, type LayoutLink, type LayoutResult, type Placement } from "./layout";

const CHARACTER_WIDTH = 6.6;
const MIN_WIDTH = 176;
const MAX_WIDTH = 320;
const MODULE_HEIGHT = 56;
const HEADER_HEIGHT = 22;
const PADDING = 16;
const GAPS = { rank: 46, node: 26 };
const LANE = 14;

export interface GraphNodeData extends Record<string, unknown> {
  id: string;
  name: string;
  moduleType: string;
  /** Wrapper names folded into this box, shown after the name. */
  chain: string[];
  hue: number;
  repeatCount: number;
  unfolded: boolean;
  parameterCount: number;
  invocationCount: number;
  container: boolean;
  open: boolean;
  /** The selected box, or one wired straight to it. */
  linked: boolean;
  /** Something is selected or searched for and this box has no part in it. */
  dimmed: boolean;
}

export interface GraphEdgeData extends Record<string, unknown> {
  /** The wire takes the hue of the block it runs inside. */
  colour: string;
  /** Lane centres the wire threads through, in flow coordinates. */
  points: Placement[];
  label: string;
  linked: boolean;
  dimmed: boolean;
}

export interface GraphView {
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
}

export interface GraphOptions {
  hues: Map<string, number>;
  selectedId: string | null;
  searching: boolean;
  showShapes: boolean;
}

export function buildGraph(view: LevelView, options: GraphOptions): GraphView {
  const paths = fullPaths(view);
  const measured = new Map<string, Measurement>();
  const size = (box: LevelNode): Measurement => {
    const cached = measured.get(box.id);
    if (cached !== undefined) return cached;
    const value = box.open ? measureContainer(box, view.edges, paths, size) : measureModule(box);
    measured.set(box.id, value);
    return value;
  };
  const root = measureChildren(view.roots, view.edges, paths, null, size);

  const nodes: Node<GraphNodeData>[] = [];
  const origins = new Map<string, Placement>();
  const highlight = linkedIds(view, options.selectedId);
  // A box drawn open is a block of its own and takes its own hue; everything closed inside one
  // wears that block's hue, which is what makes a container read as a single thing.
  const emit = (
    boxes: LevelNode[],
    parentId: string | null,
    offsets: Map<string, Placement>,
    origin: Placement,
    parentHue: number | null,
  ) => {
    for (const box of boxes) {
      const hue = box.open || parentHue === null ? (options.hues.get(box.id) as number) : parentHue;
      const measurement = size(box);
      const offset = offsets.get(box.id) as Placement;
      const absolute = { x: origin.x + offset.x, y: origin.y + offset.y };
      origins.set(box.id, absolute);
      const linked = highlight.has(box.id);
      nodes.push({
        id: box.id,
        type: box.open ? "container" : "module",
        position: offset,
        // Explicit size: containers are drawn around their children and the MiniMap needs dimensions.
        width: measurement.width,
        height: measurement.height,
        // An open container is only a frame: its body ignores the pointer so a click there does
        // not collapse everything, and only its title bar answers.
        style: { width: measurement.width, height: measurement.height, pointerEvents: box.open ? "none" : undefined },
        selected: box.id === options.selectedId,
        draggable: false,
        ...(parentId === null ? {} : { parentId, extent: "parent" as const }),
        zIndex: box.open ? 0 : 10,
        data: {
          id: box.id,
          name: box.label.name,
          moduleType: box.label.moduleType,
          chain: box.chain,
          hue,
          repeatCount: box.repeatCount,
          unfolded: box.unfolded,
          parameterCount: box.parameterCount,
          invocationCount: box.invocationCount,
          container: box.node.childIds.length > 0,
          open: box.open,
          linked,
          // Selecting fades the boxes a wire does not reach, but never the containers that
          // frame them, so the rest of the model stays readable as context.
          dimmed: options.searching ? !box.matched : highlight.size > 0 && !linked && !box.open,
        },
      });
      if (box.open) emit(box.children, box.id, measurement.offsets, absolute, hue);
    }
  };
  emit(view.roots, null, root.offsets, { x: 0, y: 0 }, null);

  const edges = view.edges.map((edge) => {
    const linked = highlight.has(edge.source) && highlight.has(edge.target);
    const holder = holderOf(edge, paths);
    const hue = holder === null ? null : (options.hues.get(holder) as number);
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: "routed",
      zIndex: linked ? 11 : 5,
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12, color: "var(--color-strong)" },
      data: {
        colour: hue === null ? "var(--color-strong)" : hueLine(hue, 70),
        points: waypoints(edge, paths, measured, root, origins),
        label: options.showShapes ? edge.shape : "",
        linked,
        dimmed: highlight.size > 0 && !linked,
      },
    };
  });
  return { nodes, edges };
}

interface Measurement {
  width: number;
  height: number;
  offsets: Map<string, Placement>;
  /** Waypoints for links between this box's children, in this box's own coordinates. */
  routes: Map<string, Placement[]>;
  /** Where each wire crossing this box's border passes through it, in the box's own coordinates. */
  ports: Map<string, Placement>;
}

function measureModule(box: LevelNode): Measurement {
  const trail = box.chain.join(" ").length + box.chain.length * 2;
  const longest = Math.max(box.label.moduleType.length, box.label.name.length + trail + 6);
  const width = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(longest * CHARACTER_WIDTH) + 92));
  return { width, height: MODULE_HEIGHT, offsets: new Map(), routes: new Map(), ports: new Map() };
}

function measureContainer(
  box: LevelNode,
  edges: LevelEdge[],
  paths: Map<string, string[]>,
  size: (box: LevelNode) => Measurement,
): Measurement {
  const inner = measureChildren(box.children, edges, paths, box.id, size);
  const trail = box.chain.join(" ").length + box.chain.length * 2;
  const label = Math.round((box.label.moduleType.length + box.label.name.length + trail) * CHARACTER_WIDTH) + 132;
  return {
    width: Math.max(inner.width + 2 * PADDING, label),
    height: inner.height + HEADER_HEIGHT + PADDING,
    offsets: inner.offsets,
    routes: inner.routes,
    ports: inner.ports,
  };
}

/** Lays the children of one container out on their own, in coordinates relative to that box. */
function measureChildren(
  children: LevelNode[],
  edges: LevelEdge[],
  paths: Map<string, string[]>,
  containerId: string | null,
  size: (box: LevelNode) => Measurement,
): Measurement {
  const boxes: LayoutBox[] = children.map((child) => {
    const measurement = size(child);
    return { id: child.id, width: measurement.width, height: measurement.height };
  });
  const { links, doorways } = innerLinks(edges, paths, containerId);
  const result = links.length === 0 ? stack(boxes) : layeredLayout([...boxes, ...doorways], links, GAPS);
  const top = containerId === null ? 0 : HEADER_HEIGHT;
  const left = containerId === null ? 0 : PADDING;
  const shift = (point: Placement) => ({ x: point.x + left, y: point.y + top });
  const drawn = new Set(boxes.map((box) => box.id));
  const offsets = new Map<string, Placement>();
  const ports = new Map<string, Placement>();
  for (const [id, point] of result.placements) {
    if (drawn.has(id)) offsets.set(id, shift(point));
    else ports.set(id, shift({ x: point.x + LANE / 2, y: point.y }));
  }
  return {
    width: result.width,
    height: result.height,
    offsets,
    ports,
    routes: new Map([...result.routes].map(([id, points]) => [id, points.map(shift)])),
  };
}

/**
 * A graph recorded without dataflow has nothing to lay out, so its modules are stacked in the
 * order they were declared, which is the order they run in.
 */
function stack(boxes: LayoutBox[]): LayoutResult {
  if (boxes.length === 0) return { placements: new Map(), routes: new Map(), width: 0, height: 0 };
  const placements = new Map<string, Placement>();
  const width = Math.max(...boxes.map((box) => box.width));
  let y = 0;
  for (const box of boxes) {
    placements.set(box.id, { x: (width - box.width) / 2, y });
    y += box.height + GAPS.rank;
  }
  return { placements, routes: new Map(), width, height: Math.max(y - GAPS.rank, 0) };
}

/**
 * Connections seen from inside one container: those running between two of its children, and those
 * crossing its border, which get a doorway of their own so the crossing has a reserved column.
 */
function innerLinks(
  edges: LevelEdge[],
  paths: Map<string, string[]>,
  containerId: string | null,
): { links: LayoutLink[]; doorways: LayoutBox[] } {
  const depth = containerId === null ? 0 : (paths.get(containerId) as string[]).length;
  const links = new Map<string, LayoutLink>();
  const doorways: LayoutBox[] = [];
  for (const edge of edges) {
    const source = childAt(paths.get(edge.source) as string[], containerId, depth);
    const target = childAt(paths.get(edge.target) as string[], containerId, depth);
    if (source === target) continue;
    if (source !== null && target !== null) {
      links.set(`${source}->${target}`, { source, target });
      continue;
    }
    if (containerId === null) continue;
    // One doorway per module and direction, not per wire: everything a module sends out of the
    // block leaves through the same column and fans out beyond the border.
    if (source !== null) {
      const port = exitPort(source);
      if (!links.has(`${source}->${port}`)) doorways.push({ id: port, width: LANE, height: 0, port: "exit" });
      links.set(`${source}->${port}`, { source, target: port });
    } else {
      const port = entryPort(target as string);
      if (!links.has(`${port}->${target}`)) doorways.push({ id: port, width: LANE, height: 0, port: "entry" });
      links.set(`${port}->${target}`, { source: port, target: target as string });
    }
  }
  return { links: [...links.values()], doorways };
}

function entryPort(childId: string): string {
  return `in:${childId}`;
}

function exitPort(childId: string): string {
  return `out:${childId}`;
}

function childAt(path: string[], containerId: string | null, depth: number): string | null {
  if (containerId !== null && path[depth - 1] !== containerId) return null;
  return path[depth] ?? null;
}

/** Every box addressed by its chain of containers, outermost first, used to project edges. */
function fullPaths(view: LevelView): Map<string, string[]> {
  const paths = new Map<string, string[]>();
  for (const [id, ancestors] of view.ancestors) paths.set(id, [...ancestors, id]);
  return paths;
}

/** The selected box plus everything one hop away from it along the dataflow. */
function linkedIds(view: LevelView, selectedId: string | null): Set<string> {
  if (selectedId === null) return new Set();
  const linked = new Set([selectedId]);
  for (const edge of view.edges) {
    if (edge.source === selectedId) linked.add(edge.target);
    if (edge.target === selectedId) linked.add(edge.source);
  }
  return linked;
}

/** The innermost container holding both ends of an edge, or null when they sit at the top level. */
function holderOf(edge: LevelEdge, paths: Map<string, string[]>): string | null {
  const source = paths.get(edge.source) as string[];
  const target = paths.get(edge.target) as string[];
  let depth = 0;
  while (depth < source.length - 1 && depth < target.length - 1 && source[depth] === target[depth]) depth += 1;
  return depth === 0 ? null : source[depth - 1];
}

/**
 * The wire's course, assembled level by level: out through each container that holds the source,
 * across the block they share, then in through each container that holds the target. Every leg is
 * a route the corresponding layout reserved, so the wire stays in its own column throughout.
 */
function waypoints(
  edge: LevelEdge,
  paths: Map<string, string[]>,
  measured: Map<string, Measurement>,
  root: Measurement,
  origins: Map<string, Placement>,
): Placement[] {
  const source = paths.get(edge.source) as string[];
  const target = paths.get(edge.target) as string[];
  let depth = 0;
  while (depth < source.length - 1 && depth < target.length - 1 && source[depth] === target[depth]) depth += 1;
  const points: Placement[] = [];
  const leg = (containerId: string, key: string, portId: string | null, before: boolean) => {
    const holder = measured.get(containerId);
    const origin = origins.get(containerId);
    if (holder === undefined || origin === undefined) return;
    const port = portId === null ? undefined : holder.ports.get(portId);
    if (before && port !== undefined) points.push(shifted(origin, port));
    for (const point of holder.routes.get(key) ?? []) points.push(shifted(origin, point));
    if (!before && port !== undefined) points.push(shifted(origin, port));
  };
  for (let level = source.length - 2; level >= depth; level -= 1) {
    const child = source[level + 1];
    leg(source[level], `${child}->${exitPort(child)}`, exitPort(child), false);
  }
  const holderId = depth === 0 ? null : source[depth - 1];
  const holder = holderId === null ? root : measured.get(holderId);
  const origin = holderId === null ? { x: 0, y: 0 } : origins.get(holderId);
  if (holder !== undefined && origin !== undefined) {
    for (const point of holder.routes.get(`${source[depth]}->${target[depth]}`) ?? []) {
      points.push(shifted(origin, point));
    }
  }
  for (let level = depth; level <= target.length - 2; level += 1) {
    const child = target[level + 1];
    leg(target[level], `${entryPort(child)}->${child}`, entryPort(child), true);
  }
  return points;
}

function shifted(origin: Placement, point: Placement): Placement {
  return { x: origin.x + point.x, y: origin.y + point.y };
}
