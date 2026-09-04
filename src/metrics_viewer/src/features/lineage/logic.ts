import type { Checkpoint, Lineage, LineageRun } from "@/shared/types";

import { runChains } from "./chains";

export const NODE_WIDTH = 132;
export const NODE_HEIGHT = 58;
/** Wide enough to hold the fork channels routed between two columns. */
export const COLUMN_GAP = 84;
export const LANE_HEIGHT = 116;
export const BOX_PADDING_X = 16;
export const BOX_PADDING_Y = 30;
export const COLUMN_PITCH = NODE_WIDTH + COLUMN_GAP;
/** First channel sits this far right of a node, and channels are this far apart. */
const CHANNEL_INSET = 20;
const CHANNEL_STEP = 14;
/** Corner radius of the orthogonal fork path. */
const CORNER_RADIUS = 10;

export interface PlacedCheckpoint extends Checkpoint {
  /** Column: how many checkpoints deep in the resume chain this one sits. */
  depth: number;
  /** Completed steps including the runs this checkpoint's run resumed from. */
  lineageStep: number;
  /** Row: one lane per run. */
  lane: number;
  x: number;
  y: number;
}

export interface RunBox {
  runId: string;
  name: string;
  run: LineageRun | undefined;
  /** Run this one forked from, or null when it starts from scratch. */
  parentRunId: string | null;
  lane: number;
  x: number;
  y: number;
  width: number;
  height: number;
  checkpoints: PlacedCheckpoint[];
}

export interface LineageLink {
  id: string;
  source: string;
  target: string;
  /** A link between two runs — the fork drawn from the resumed checkpoint. */
  fork: boolean;
  runId: string;
  /**
   * Absolute x of the vertical segment a fork travels along. Two forks that pass the same
   * rows get different channels, so their lines never sit on top of each other.
   */
  channelX: number;
}

export interface LineageLayout {
  nodes: PlacedCheckpoint[];
  boxes: RunBox[];
  links: LineageLink[];
}

/** Depth of every checkpoint in its resume chain; tolerates missing and cyclic ancestors. */
function depths(byId: Map<string, Checkpoint>): Map<string, number> {
  const result = new Map<string, number>();
  const resolve = (checkpoint: Checkpoint, seen: Set<string>): number => {
    const cached = result.get(checkpoint.id);
    if (cached !== undefined) return cached;
    const ancestor = checkpoint.ancestorId === null ? undefined : byId.get(checkpoint.ancestorId);
    const depth = ancestor === undefined || seen.has(ancestor.id) ? 0 : resolve(ancestor, seen.add(checkpoint.id)) + 1;
    result.set(checkpoint.id, depth);
    return depth;
  };
  for (const checkpoint of byId.values()) resolve(checkpoint, new Set());
  return result;
}

interface Interval {
  /** Inclusive column range the run's checkpoints cover. */
  first: number;
  last: number;
}

const overlaps = (left: Interval, right: Interval) => left.first <= right.last && right.first <= left.last;

/** Lanes to try for a child, nearest to its parent first: same row, then one below, one above, … */
function* candidates(parentLane: number): Generator<number> {
  yield parentLane;
  for (let step = 1; ; step += 1) {
    yield parentLane + step;
    yield parentLane - step;
  }
}

/**
 * One row per run, packed like a commit graph: a run is placed in the row nearest to the
 * run it resumed from that its columns are still free in — the same row when the parent
 * had already finished by then, so a plain continuation reads as one straight line, and
 * only a genuine branch is pushed off to a neighbouring row.
 *
 * Runs are placed shallowest fork first, so the branches that leave early sit closest to
 * the trunk and later ones stack outwards from it.
 */
function lanes(
  order: string[],
  parentRun: Map<string, string | null>,
  forkDepth: Map<string, number>,
  columns: Map<string, Interval>,
): Map<string, number> {
  const rank = new Map(order.map((runId, index) => [runId, index]));
  const queue = [...order].sort(
    (left, right) => (forkDepth.get(left) ?? 0) - (forkDepth.get(right) ?? 0) || (rank.get(left) ?? 0) - (rank.get(right) ?? 0),
  );

  const result = new Map<string, number>();
  const occupied = new Map<number, Interval[]>();
  const span = (runId: string) => columns.get(runId) ?? { first: 0, last: 0 };
  const free = (lane: number, runId: string) =>
    !(occupied.get(lane) ?? []).some((other) => overlaps(other, span(runId)));

  const place = (runId: string, lane: number) => {
    occupied.set(lane, [...(occupied.get(lane) ?? []), span(runId)]);
    result.set(runId, lane);
  };

  const bottom = () => (result.size === 0 ? 0 : Math.max(...result.values()) + 1);
  const placeNear = (runId: string, parentLane: number) => {
    for (const lane of candidates(parentLane)) {
      if (free(lane, runId)) {
        place(runId, lane);
        return;
      }
    }
  };

  // One tree at a time, parents before children, so a child always has a row to be near;
  // the fork order above decides which of two children of the same run gets the nearer row.
  let remaining = queue.filter((runId) => (parentRun.get(runId) ?? null) !== null);
  const grow = () => {
    for (let progress = true; progress; ) {
      progress = false;
      remaining = remaining.filter((runId) => {
        const parentLane = result.get(parentRun.get(runId) as string);
        if (parentLane === undefined) return true;
        placeNear(runId, parentLane);
        progress = true;
        return false;
      });
    }
  };
  for (const runId of queue.filter((item) => (parentRun.get(item) ?? null) === null)) {
    place(runId, bottom());
    grow();
  }
  // Whatever is left resumed from a run outside the project, or sits in a cycle.
  while (remaining.length > 0) {
    const orphan = remaining[0];
    remaining = remaining.slice(1);
    place(orphan, bottom());
    grow();
  }

  // Rows may have grown upwards; shift them back so the topmost run sits at lane 0.
  const top = Math.min(...result.values());
  return new Map([...result].map(([runId, lane]) => [runId, lane - top]));
}

interface Span {
  top: number;
  bottom: number;
}

/**
 * One free channel per fork, per gap between two columns. Forks whose vertical segments
 * would run over the same rows are pushed onto separate channels; forks that cannot
 * collide reuse the first one, so a shallow graph stays on a single tidy trunk.
 */
function assignChannels(forks: { link: LineageLink; gap: number; span: Span }[]): void {
  const taken = new Map<number, Span[][]>();
  const ordered = [...forks].sort((left, right) => (right.span.bottom - right.span.top) - (left.span.bottom - left.span.top));
  for (const fork of ordered) {
    const channels = taken.get(fork.gap) ?? [];
    let index = channels.findIndex(
      (spans) => !spans.some((span) => fork.span.top < span.bottom && span.top < fork.span.bottom),
    );
    if (index === -1) {
      index = channels.length;
      channels.push([]);
    }
    channels[index].push(fork.span);
    taken.set(fork.gap, channels);
    // Wrap once the gap is full rather than drawing on top of the next column of nodes.
    const room = Math.max(1, Math.floor((COLUMN_GAP - CHANNEL_INSET - 8) / CHANNEL_STEP));
    fork.link.channelX = fork.gap * COLUMN_PITCH + NODE_WIDTH + CHANNEL_INSET + (index % room) * CHANNEL_STEP;
  }
}

export function layoutLineage(lineage: Lineage, runNames: Map<string, string>): LineageLayout {
  const byId = new Map(lineage.checkpoints.map((checkpoint) => [checkpoint.id, checkpoint]));
  const depth = depths(byId);
  const ordered = [...lineage.checkpoints].sort(
    (left, right) => (depth.get(left.id) ?? 0) - (depth.get(right.id) ?? 0) || left.createdAt - right.createdAt,
  );

  const runOrder: string[] = [];
  const runCheckpoints = new Map<string, Checkpoint[]>();
  for (const checkpoint of ordered) {
    const own = runCheckpoints.get(checkpoint.runId);
    if (own === undefined) {
      runOrder.push(checkpoint.runId);
      runCheckpoints.set(checkpoint.runId, [checkpoint]);
    } else own.push(checkpoint);
  }

  // The run graph is derived once in ./chains, so the panel and the charts agree on which
  // run resumed from which.
  const chains = runChains(lineage);
  const parentRun = new Map(runOrder.map((runId) => [runId, chains.get(runId)?.parentRunId ?? null]));
  const forkDepth = new Map(runOrder.map((runId) => {
    const forkId = chains.get(runId)?.forkCheckpointId ?? null;
    return [runId, forkId === null ? 0 : depth.get(forkId) ?? 0];
  }));

  const columns = new Map(runOrder.map((runId) => {
    const own = (runCheckpoints.get(runId) ?? []).map((checkpoint) => depth.get(checkpoint.id) ?? 0);
    return [runId, { first: Math.min(...own), last: Math.max(...own) }];
  }));
  const lane = lanes(runOrder, parentRun, forkDepth, columns);
  const nodes: PlacedCheckpoint[] = ordered.map((checkpoint) => {
    const column = depth.get(checkpoint.id) ?? 0;
    const row = lane.get(checkpoint.runId) ?? 0;
    return {
      ...checkpoint,
      depth: column,
      lineageStep: checkpoint.step + (chains.get(checkpoint.runId)?.offset ?? 0),
      lane: row,
      x: column * COLUMN_PITCH,
      y: row * LANE_HEIGHT,
    };
  });

  const placedById = new Map(nodes.map((node) => [node.id, node]));
  const boxes: RunBox[] = runOrder.map((runId) => {
    const own = nodes.filter((node) => node.runId === runId);
    const left = Math.min(...own.map((node) => node.x));
    const right = Math.max(...own.map((node) => node.x + NODE_WIDTH));
    const top = Math.min(...own.map((node) => node.y));
    const bottom = Math.max(...own.map((node) => node.y + NODE_HEIGHT));
    return {
      runId,
      name: runNames.get(runId) ?? runId.slice(0, 8),
      run: lineage.runs.find((item) => item.id === runId),
      parentRunId: parentRun.get(runId) ?? null,
      lane: lane.get(runId) ?? 0,
      x: left - BOX_PADDING_X,
      y: top - BOX_PADDING_Y,
      width: right - left + BOX_PADDING_X * 2,
      height: bottom - top + BOX_PADDING_Y + BOX_PADDING_X,
      checkpoints: own,
    };
  });

  const links: LineageLink[] = [];
  const forks: { link: LineageLink; gap: number; span: Span }[] = [];
  for (const node of nodes) {
    if (node.ancestorId === null) continue;
    const ancestor = placedById.get(node.ancestorId);
    if (ancestor === undefined) continue;
    const fork = ancestor.runId !== node.runId || ancestor.lane !== node.lane;
    const link: LineageLink = {
      id: `${ancestor.id}-${node.id}`,
      source: ancestor.id,
      target: node.id,
      fork,
      runId: node.runId,
      channelX: ancestor.x + NODE_WIDTH + COLUMN_GAP / 2,
    };
    links.push(link);
    if (fork) {
      const top = Math.min(ancestor.y, node.y) + NODE_HEIGHT / 2;
      const bottom = Math.max(ancestor.y, node.y) + NODE_HEIGHT / 2;
      forks.push({ link, gap: ancestor.depth, span: { top, bottom } });
    }
  }
  assignChannels(forks);

  return { nodes, boxes, links };
}

/**
 * Orthogonal path for a fork: out of the source, down (or up) a dedicated channel, then
 * into the target. Corners are rounded so a branch reads as one line, not three.
 */
export function forkPath(
  sourceX: number,
  sourceY: number,
  channelX: number,
  targetX: number,
  targetY: number,
): string {
  const down = targetY >= sourceY ? 1 : -1;
  const lane = Math.max(sourceX + 2, Math.min(channelX, targetX - 2));
  const radius = Math.min(CORNER_RADIUS, Math.abs(targetY - sourceY) / 2, Math.abs(lane - sourceX), Math.abs(targetX - lane));
  if (radius <= 1) return `M ${sourceX},${sourceY} L ${lane},${sourceY} L ${lane},${targetY} L ${targetX},${targetY}`;
  return [
    `M ${sourceX},${sourceY}`,
    `L ${lane - radius},${sourceY}`,
    `Q ${lane},${sourceY} ${lane},${sourceY + radius * down}`,
    `L ${lane},${targetY - radius * down}`,
    `Q ${lane},${targetY} ${lane + radius},${targetY}`,
    `L ${targetX},${targetY}`,
  ].join(" ");
}

/** Checkpoint ids on the path from a root down to `id`, oldest first. */
export function ancestorChain(id: string, nodes: PlacedCheckpoint[]): PlacedCheckpoint[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const chain: PlacedCheckpoint[] = [];
  let current = byId.get(id);
  while (current !== undefined && !chain.some((node) => node.id === current?.id)) {
    chain.unshift(current);
    current = current.ancestorId === null ? undefined : byId.get(current.ancestorId);
  }
  return chain;
}

export function matchesSearch(box: RunBox, search: string): boolean {
  if (search.length === 0) return true;
  const needle = search.toLowerCase();
  return box.name.toLowerCase().includes(needle)
    || box.checkpoints.some((checkpoint) => checkpoint.name.toLowerCase().includes(needle));
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} kB`;
  return `${bytes} B`;
}
