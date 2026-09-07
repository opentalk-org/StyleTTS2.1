import type { ModelTree, TreeNode } from "./hierarchy";
import type { RepeatIndex } from "./repeats";

/**
 * One box on the canvas. A box is either open, with its children nested inside it, or closed and
 * standing in for everything below it. Either way it can cover several structurally identical
 * instances at once — the twelve blocks of a stack are drawn once, marked with their count.
 */
export interface LevelNode {
  id: string;
  node: TreeNode;
  /** What the box is titled with: the outermost module folded into it, if any. */
  label: { moduleType: string; name: string };
  /** Names of the modules that title swallowed, outermost first. */
  chain: string[];
  /** Identical blocks in this run; above 1 only on the head of a repeat run. */
  repeatCount: number;
  /** The run is drawn block by block instead of folded into this one box. */
  unfolded: boolean;
  open: boolean;
  children: LevelNode[];
  parameterCount: number;
  invocationCount: number;
  matched: boolean;
}

export interface LevelEdge {
  id: string;
  source: string;
  target: string;
  shape: string;
  /** Recorded connections merged into this one, which is the repeat count on a folded box. */
  count: number;
}

export interface LevelView {
  roots: LevelNode[];
  edges: LevelEdge[];
  /** Box id to the chain of box ids containing it, outermost first. */
  ancestors: Map<string, string[]>;
  boxCount: number;
}

export function groupKey(id: string): string {
  return `group:${id}`;
}

/** Which repetition of which stack an invocation was claimed through, so folded repeats do not
 *  grow edges between blocks that are drawn as one. */
interface ClaimKey {
  group: string;
  member: string;
}

interface Claim {
  box: string;
  key: ClaimKey | null;
}

export interface LevelOptions {
  expanded: Set<string>;
  matches: Set<string>;
  /** Hide anything whose subtree holds no weights, wiring its neighbours straight together. */
  parametersOnly: boolean;
}

export function levelView(tree: ModelTree, repeats: RepeatIndex, options: LevelOptions): LevelView {
  const { expanded, matches } = options;
  const claims = new Map<string, Claim>();
  const ancestors = new Map<string, string[]>();
  let boxCount = 0;

  const box = (
    instances: string[],
    keys: (ClaimKey | null)[],
    repeatCount: number,
    unfolded: boolean,
    path: string[],
    open: boolean,
  ) => {
    const id = instances[0];
    ancestors.set(id, path);
    boxCount += 1;
    const nodes = instances.map((instanceId) => tree.nodes.get(instanceId) as TreeNode);
    let matched = false;
    if (!open) {
      for (const [index, instanceId] of instances.entries()) {
        for (const claimedId of descendants(tree, instanceId)) {
          claims.set(claimedId, { box: id, key: keys[index] });
          matched = matched || matches.has(claimedId);
        }
      }
    }
    return {
      id,
      node: nodes[0],
      label: { moduleType: nodes[0].moduleType, name: nodes[0].name },
      chain: [] as string[],
      repeatCount,
      unfolded,
      open,
      children: [] as LevelNode[],
      parameterCount: parametersOf(nodes),
      invocationCount: nodes.reduce((total, node) => total + node.invocationCount, 0),
      matched,
    };
  };

  /** `siblings[i]` holds the children of instance `i` of one module, aligned index by index. */
  const build = (siblings: string[][], keys: (ClaimKey | null)[], path: string[]): LevelNode[] => {
    const boxes: LevelNode[] = [];
    const folded = new Set<string>();
    for (const [index, primary] of siblings[0].entries()) {
      if (folded.has(primary)) continue;
      const group = repeats.groups.get(primary);
      const unfolded = group !== undefined && expanded.has(groupKey(primary));
      // Members are addressed by position, so the same fold applies to every mirrored instance.
      const positions = group === undefined || unfolded
        ? [index]
        : [...siblings[0].entries()].filter(([, id]) => repeats.groupOf.get(id) === primary || id === primary).map(([at]) => at);
      for (const at of positions) folded.add(siblings[0][at]);
      const instances = siblings.flatMap((list) => positions.map((at) => list[at]));
      const instanceKeys = siblings.flatMap((list, instance) =>
        positions.map((at) =>
          keys[instance] ?? (positions.length > 1 ? { group: primary, member: list[at] } : null),
        ),
      );
      const node = tree.nodes.get(primary) as TreeNode;
      const open = node.childIds.length > 0 && expanded.has(primary);
      const repeatCount = group === undefined ? 1 : group.memberIds.length;
      const created = box(instances, instanceKeys, repeatCount, unfolded, path, open);
      if (open) {
        created.children = build(
          instances.map((instanceId) => (tree.nodes.get(instanceId) as TreeNode).childIds),
          instanceKeys,
          [...path, primary],
        );
        created.matched = matches.has(primary) || created.children.some((child) => child.matched);
      }
      boxes.push(created);
    }
    return boxes;
  };

  const roots = build([tree.rootIds], [null], []);
  const view = { roots, edges: liftEdges(tree, claims), ancestors, boxCount };
  return foldWrappers(options.parametersOnly ? withoutUnparameterised(view) : view);
}

/**
 * Drops every box that holds no weights and reconnects what it stood between, so activations and
 * traced tensor operations stop crowding out the layers that actually learn.
 */
function withoutUnparameterised(view: LevelView): LevelView {
  const dropped = new Set<string>();
  const keep = (boxes: LevelNode[]): LevelNode[] =>
    boxes.filter((box) => {
      box.children = keep(box.children);
      if (box.parameterCount > 0) return true;
      dropped.add(box.id);
      view.ancestors.delete(box.id);
      return false;
    });
  const roots = keep(view.roots);
  return { roots, edges: bypass(view.edges, dropped), ancestors: view.ancestors, boxCount: view.boxCount - dropped.size };
}

/** Replaces every path that runs through dropped boxes with the direct link it stands for. */
function bypass(edges: LevelEdge[], dropped: Set<string>): LevelEdge[] {
  const outgoing = new Map<string, LevelEdge[]>();
  for (const edge of edges) outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge]);
  const merged = new Map<string, LevelEdge>();
  for (const edge of edges) {
    if (dropped.has(edge.source)) continue;
    const seen = new Set<string>();
    const pending = [edge];
    while (pending.length > 0) {
      const step = pending.pop() as LevelEdge;
      if (!dropped.has(step.target)) {
        if (step.target === edge.source) continue;
        const id = `${edge.source}->${step.target}`;
        const existing = merged.get(id);
        if (existing === undefined) {
          merged.set(id, { id, source: edge.source, target: step.target, shape: edge.shape, count: edge.count });
        } else existing.count += edge.count;
        continue;
      }
      if (seen.has(step.target)) continue;
      seen.add(step.target);
      pending.push(...(outgoing.get(step.target) ?? []));
    }
  }
  return [...merged.values()];
}

/**
 * A container whose only content is one box adds a frame and says nothing, so the two are drawn as
 * a single box titled by the outer module, with the names it swallowed kept as a trail.
 */
function foldWrappers(view: LevelView): LevelView {
  const replaced = new Map<string, string>();
  const fold = (boxes: LevelNode[]): LevelNode[] =>
    boxes.map((box) => {
      box.children = fold(box.children);
      if (!box.open || box.children.length !== 1) return box;
      const [child] = box.children;
      replaced.set(box.id, child.id);
      return { ...child, label: box.label, chain: [child.label.name, ...child.chain] };
    });
  const roots = fold(view.roots);
  const settle = (id: string): string => {
    let at = id;
    while (replaced.has(at)) at = replaced.get(at) as string;
    return at;
  };
  const ancestors = new Map<string, string[]>();
  const walk = (boxes: LevelNode[], path: string[]) => {
    for (const box of boxes) {
      ancestors.set(box.id, path);
      walk(box.children, [...path, box.id]);
    }
  };
  walk(roots, []);
  const edges = view.edges.map((edge) => {
    const source = settle(edge.source);
    const target = settle(edge.target);
    return { ...edge, id: `${source}->${target}`, source, target };
  });
  return { roots, edges: edges.filter((edge) => edge.source !== edge.target), ancestors, boxCount: ancestors.size };
}

/** Repeated invocations of one module share its weights, so every module path counts once. */
function parametersOf(nodes: TreeNode[]): number {
  const counted = new Map(nodes.map((node) => [node.modulePath, node.parameterCount]));
  return [...counted.values()].reduce((total, count) => total + count, 0);
}

function descendants(tree: ModelTree, id: string): string[] {
  const node = tree.nodes.get(id) as TreeNode;
  return [id, ...node.childIds.flatMap((childId) => descendants(tree, childId))];
}

/**
 * Redirects every recorded connection onto the boxes now standing for its endpoints. Wires join the
 * modules themselves and cross block borders on the way, which is what makes a block look connected
 * to what feeds it. A connection between two repetitions of one folded block is dropped: both ends
 * are drawn by the same block, so keeping it would invent a loop the model does not have here.
 */
function liftEdges(tree: ModelTree, claims: Map<string, Claim>): LevelEdge[] {
  const merged = new Map<string, LevelEdge>();
  for (const edge of tree.edges) {
    const source = claims.get(edge.source);
    const target = claims.get(edge.target);
    if (source === undefined || target === undefined || source.box === target.box) continue;
    if (crossesRepetitions(source.key, target.key)) continue;
    const from = source.box;
    const to = target.box;
    const id = `${from}->${to}`;
    const existing = merged.get(id);
    if (existing === undefined) merged.set(id, { id, source: from, target: to, shape: edge.shape, count: 1 });
    else {
      existing.count += 1;
      if (existing.shape !== edge.shape) existing.shape = "";
    }
  }
  return [...merged.values()];
}

function crossesRepetitions(source: ClaimKey | null, target: ClaimKey | null): boolean {
  if (source === null || target === null) return false;
  return source.group === target.group && source.member !== target.member;
}

/**
 * Containers to open so that every search match is on screen. A match inside a folded repeat run
 * is opened through the block that stands for the run, which already claims it, so the run stays
 * folded instead of exploding into one copy of the hit per repetition.
 */
export function revealPaths(tree: ModelTree, repeats: RepeatIndex, matches: Set<string>): Set<string> {
  const opened = new Set<string>();
  for (const matchId of matches) {
    let ancestorId = (tree.nodes.get(matchId) as TreeNode).parentId;
    while (ancestorId !== null && !opened.has(ancestorId)) {
      opened.add(repeats.groupOf.get(ancestorId) ?? ancestorId);
      opened.add(ancestorId);
      ancestorId = (tree.nodes.get(ancestorId) as TreeNode).parentId;
    }
  }
  return opened;
}

export function matchingIds(tree: ModelTree, search: string): Set<string> {
  const needle = search.trim().toLowerCase();
  if (needle.length === 0) return new Set();
  const matches = new Set<string>();
  for (const node of tree.nodes.values()) {
    if (node.id.toLowerCase().includes(needle) || node.moduleType.toLowerCase().includes(needle)) matches.add(node.id);
  }
  return matches;
}

/** Every container down to `depth`, which is what the depth control draws open. */
export function containersToDepth(tree: ModelTree, depth: number): Set<string> {
  const opened = new Set<string>();
  for (const node of tree.nodes.values()) {
    if (node.childIds.length > 0 && node.depth < depth) opened.add(node.id);
  }
  return opened;
}
