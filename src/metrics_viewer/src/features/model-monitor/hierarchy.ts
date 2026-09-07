import type { ModelComponent } from "@/shared/types";

/** One module of the recorded graph: either an executed invocation or a container of them. */
export interface TreeNode {
  id: string;
  parentId: string | null;
  name: string;
  moduleType: string;
  /** Stable path shared by every invocation of the same module; empty for traced operations. */
  modulePath: string;
  /** True for a recorded forward invocation, which is what dataflow edges connect. */
  executed: boolean;
  parameterCount: number;
  childIds: string[];
  depth: number;
  /** Executed invocations in this subtree, including the node itself. */
  invocationCount: number;
  component: ModelComponent | null;
}

export interface DataflowEdge {
  source: string;
  target: string;
  shape: string;
}

export interface ModelTree {
  nodes: Map<string, TreeNode>;
  rootIds: string[];
  edges: DataflowEdge[];
  maxDepth: number;
}

export function buildTree(components: ModelComponent[]): ModelTree {
  const nodes = new Map<string, TreeNode>();
  const order = new Map<string, number>();
  for (const [index, component] of components.entries()) {
    const executed = component.input_ids !== undefined;
    nodes.set(component.id, {
      id: component.id,
      parentId: parentOf(component),
      name: component.name,
      moduleType: component.module_type,
      modulePath: component.module_path ?? component.id,
      executed,
      parameterCount: component.parameter_count,
      childIds: [],
      depth: 0,
      invocationCount: executed ? 1 : 0,
      component,
    });
    if (executed) order.set(component.id, index);
  }
  addMissingContainers(nodes);
  linkChildren(nodes);
  const rootIds = [...nodes.values()].filter((node) => node.parentId === null).map((node) => node.id);
  sortChildren(nodes, rootIds, order);
  const maxDepth = measure(nodes, rootIds);
  return { nodes, rootIds, edges: dataflow(components), maxDepth };
}

/**
 * Graphs recorded before container modules were emitted only carry executed leaves, so the
 * intermediate module paths they imply are materialised here. Their class name is unknown, which
 * is why `moduleType` stays empty and the box falls back to the path segment.
 */
function addMissingContainers(nodes: Map<string, TreeNode>) {
  for (const node of [...nodes.values()]) {
    let parentId = node.parentId;
    while (parentId !== null && !nodes.has(parentId)) {
      nodes.set(parentId, {
        id: parentId,
        parentId: containerParent(parentId),
        name: parentId.slice(parentId.lastIndexOf(".") + 1),
        moduleType: "",
        modulePath: parentId,
        executed: false,
        parameterCount: 0,
        childIds: [],
        depth: 0,
        invocationCount: 0,
        component: null,
      });
      parentId = containerParent(parentId);
    }
  }
}

function linkChildren(nodes: Map<string, TreeNode>) {
  for (const node of nodes.values()) {
    if (node.parentId === null) continue;
    (nodes.get(node.parentId) as TreeNode).childIds.push(node.id);
  }
}

/** Children read top to bottom in execution order; a container sorts by its earliest invocation. */
function sortChildren(nodes: Map<string, TreeNode>, rootIds: string[], order: Map<string, number>) {
  const rank = new Map<string, number>();
  const rankOf = (id: string): number => {
    const cached = rank.get(id);
    if (cached !== undefined) return cached;
    const node = nodes.get(id) as TreeNode;
    const own = order.get(id) ?? Number.MAX_SAFE_INTEGER;
    const value = node.childIds.reduce((least, childId) => Math.min(least, rankOf(childId)), own);
    rank.set(id, value);
    return value;
  };
  for (const id of nodes.keys()) rankOf(id);
  for (const node of nodes.values()) {
    node.childIds.sort((left, right) => (rank.get(left) as number) - (rank.get(right) as number));
  }
  rootIds.sort((left, right) => (rank.get(left) as number) - (rank.get(right) as number));
}

/** Depth and subtree totals in one post-order walk. */
function measure(nodes: Map<string, TreeNode>, rootIds: string[]): number {
  let maxDepth = 0;
  const visit = (id: string, depth: number) => {
    const node = nodes.get(id) as TreeNode;
    node.depth = depth;
    maxDepth = Math.max(maxDepth, depth);
    for (const childId of node.childIds) visit(childId, depth + 1);
    node.invocationCount = node.childIds.reduce(
      (total, childId) => total + (nodes.get(childId) as TreeNode).invocationCount,
      node.executed ? 1 : 0,
    );
    // A recorded container carries its own recursive total, but a graph that only counts leaves
    // reports zero for it, so the larger of the two readings is the honest one.
    const summed = node.childIds.reduce(
      (total, childId) => total + (nodes.get(childId) as TreeNode).parameterCount,
      0,
    );
    node.parameterCount = Math.max(node.parameterCount, summed);
  };
  for (const id of rootIds) visit(id, 0);
  return maxDepth;
}

function dataflow(components: ModelComponent[]): DataflowEdge[] {
  const shapes = new Map(components.map((component) => [component.id, component.output_shapes?.[0] ?? ""]));
  return components.flatMap((component) =>
    (component.input_ids ?? [])
      .filter((source) => shapes.has(source))
      .map((source) => ({ source, target: component.id, shape: shapes.get(source) as string })),
  );
}

function parentOf(component: ModelComponent): string | null {
  if (component.parent_id !== null) return component.parent_id;
  const path = component.module_path;
  if (path === undefined || path === "") return null;
  return containerParent(path);
}

function containerParent(path: string): string | null {
  const cut = path.lastIndexOf(".");
  return cut === -1 ? null : path.slice(0, cut);
}

export function formatParameterCount(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(2)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
  return String(count);
}
