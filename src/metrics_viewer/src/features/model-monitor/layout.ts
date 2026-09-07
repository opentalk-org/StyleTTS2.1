export interface LayoutBox {
  id: string;
  width: number;
  height: number;
  /** A doorway in the container's border rather than a drawn box: where a wire enters or leaves. */
  port?: "entry" | "exit";
}

export interface LayoutLink {
  source: string;
  target: string;
}

export interface Placement {
  x: number;
  y: number;
}

export interface LayoutResult {
  placements: Map<string, Placement>;
  /**
   * One entry per gap a link crosses, keyed `source->target`: the y of that gap and the x the
   * wire sits at once through it. Following them keeps every sideways run between two ranks.
   */
  routes: Map<string, Placement[]>;
  width: number;
  height: number;
}

export interface LayoutGaps {
  rank: number;
  node: number;
}

const SWEEPS = 6;
/** Width reserved for one rank-skipping link, so its wire gets a corridor of its own. */
const LANE = 14;

/**
 * Layered top-to-bottom placement: rank by longest path, then reduce crossings and pull each
 * node towards its neighbours. Dummy nodes stand in for edges that skip ranks so that residual
 * connections get a lane of their own instead of cutting across the boxes between them.
 */
export function layeredLayout(boxes: LayoutBox[], links: LayoutLink[], gaps: LayoutGaps): LayoutResult {
  if (boxes.length === 0) return { placements: new Map(), routes: new Map(), width: 0, height: 0 };
  const present = new Set(boxes.map((box) => box.id));
  const ports = new Map(boxes.filter((box) => box.port !== undefined).map((box) => [box.id, box.port as string]));
  const drawn = links.filter((link) => present.has(link.source) && present.has(link.target) && link.source !== link.target);
  const acyclic = removeCycles(boxes, drawn);
  const ranks = portRanks(boxes, rankNodes(boxes, acyclic.filter((link) => !ports.has(link.source) && !ports.has(link.target))));
  const layers = buildLayers(boxes, acyclic, ranks);
  orderLayers(layers, acyclic);
  return placeLayers(layers, acyclic, drawn, gaps);
}

/** Depth-first back edges are dropped for ranking only; they are still drawn on the canvas. */
function removeCycles(boxes: LayoutBox[], links: LayoutLink[]): LayoutLink[] {
  const outgoing = new Map<string, string[]>(boxes.map((box) => [box.id, []]));
  for (const link of links) {
    if (link.source !== link.target) (outgoing.get(link.source) as string[]).push(link.target);
  }
  const state = new Map<string, number>();
  const back = new Set<string>();
  const visit = (id: string) => {
    state.set(id, 1);
    for (const target of outgoing.get(id) as string[]) {
      const seen = state.get(target);
      if (seen === 1) back.add(`${id}->${target}`);
      else if (seen === undefined) visit(target);
    }
    state.set(id, 2);
  };
  for (const box of boxes) if (state.get(box.id) === undefined) visit(box.id);
  return links.filter((link) => link.source !== link.target && !back.has(`${link.source}->${link.target}`));
}

function rankNodes(boxes: LayoutBox[], links: LayoutLink[]): Map<string, number> {
  const incoming = new Map<string, string[]>(boxes.map((box) => [box.id, []]));
  const outgoing = new Map<string, string[]>(boxes.map((box) => [box.id, []]));
  for (const link of links) {
    (incoming.get(link.target) as string[]).push(link.source);
    (outgoing.get(link.source) as string[]).push(link.target);
  }
  const remaining = new Map(boxes.map((box) => [box.id, (incoming.get(box.id) as string[]).length]));
  const ranks = new Map(boxes.map((box) => [box.id, 0]));
  const ready = boxes.filter((box) => remaining.get(box.id) === 0).map((box) => box.id);
  while (ready.length > 0) {
    const id = ready.shift() as string;
    const rank = ranks.get(id) as number;
    for (const target of outgoing.get(id) as string[]) {
      ranks.set(target, Math.max(ranks.get(target) as number, rank + 1));
      const left = (remaining.get(target) as number) - 1;
      remaining.set(target, left);
      if (left === 0) ready.push(target);
    }
  }
  return ranks;
}

/**
 * Doorways sit in rows of their own above and below the drawn ranks, so a wire crossing the border
 * has a column reserved for it all the way through the block instead of cutting across its boxes.
 */
function portRanks(boxes: LayoutBox[], ranks: Map<string, number>): Map<string, number> {
  const drawn = boxes.filter((box) => box.port === undefined);
  const deepest = Math.max(...drawn.map((box) => ranks.get(box.id) as number), 0);
  const placed = new Map<string, number>();
  for (const box of boxes) {
    if (box.port === "entry") placed.set(box.id, 0);
    else if (box.port === "exit") placed.set(box.id, deepest + 2);
    else placed.set(box.id, (ranks.get(box.id) as number) + 1);
  }
  return placed;
}

interface LayerNode {
  id: string;
  width: number;
  height: number;
  x: number;
  /** A stand-in for a link crossing this rank rather than a box that is drawn. */
  lane: boolean;
}

function buildLayers(boxes: LayoutBox[], links: LayoutLink[], ranks: Map<string, number>): LayerNode[][] {
  const depth = Math.max(...boxes.map((box) => ranks.get(box.id) as number)) + 1;
  const layers: LayerNode[][] = Array.from({ length: depth }, () => []);
  for (const box of boxes) {
    layers[ranks.get(box.id) as number].push({ id: box.id, width: box.width, height: box.height, x: 0, lane: false });
  }
  for (const link of links) {
    const from = ranks.get(link.source) as number;
    const to = ranks.get(link.target) as number;
    for (let rank = from + 1; rank < to; rank += 1) {
      layers[rank].push({ id: dummyId(link, rank), width: LANE, height: 0, x: 0, lane: true });
    }
  }
  return layers;
}

function dummyId(link: LayoutLink, rank: number): string {
  return `~${link.source}~${link.target}~${rank}`;
}

/** Median heuristic, swept downwards and upwards until the order settles. */
function orderLayers(layers: LayerNode[][], links: LayoutLink[]) {
  const { above, below } = segments(layers, links);
  for (let sweep = 0; sweep < SWEEPS; sweep += 1) {
    const downward = sweep % 2 === 0;
    const range = downward ? [...layers.keys()] : [...layers.keys()].reverse();
    for (const index of range) {
      const neighbours = downward ? above : below;
      const positions = positionIndex(layers, downward ? index - 1 : index + 1);
      if (positions === null) continue;
      const scored = layers[index].map((node, order) => {
        const value = median((neighbours.get(node.id) ?? []).map((id) => positions.get(id) as number));
        return { node, order, median: value ?? order };
      });
      scored.sort((left, right) => left.median - right.median || left.order - right.order);
      layers[index] = scored.map((entry) => entry.node);
    }
  }
}

/** Neighbour lists split by direction, with rank-skipping edges routed through their dummies. */
function segments(layers: LayerNode[][], links: LayoutLink[]) {
  const rankOf = new Map<string, number>();
  for (const [rank, layer] of layers.entries()) for (const node of layer) rankOf.set(node.id, rank);
  const above = new Map<string, string[]>();
  const below = new Map<string, string[]>();
  const connect = (upper: string, lower: string) => {
    if (!rankOf.has(upper) || !rankOf.has(lower)) return;
    const incoming = above.get(lower);
    if (incoming === undefined) above.set(lower, [upper]);
    else incoming.push(upper);
    const outgoing = below.get(upper);
    if (outgoing === undefined) below.set(upper, [lower]);
    else outgoing.push(lower);
  };
  for (const link of links) {
    const from = rankOf.get(link.source) as number;
    const to = rankOf.get(link.target) as number;
    const chain = [link.source];
    for (let rank = from + 1; rank < to; rank += 1) chain.push(dummyId(link, rank));
    chain.push(link.target);
    for (let index = 0; index + 1 < chain.length; index += 1) connect(chain[index], chain[index + 1]);
  }
  return { above, below };
}

function positionIndex(layers: LayerNode[][], index: number): Map<string, number> | null {
  if (index < 0 || index >= layers.length) return null;
  return new Map(layers[index].map((node, order) => [node.id, order]));
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Nodes are pulled to the centre of their neighbours, then pushed apart within their layer. */
function placeLayers(layers: LayerNode[][], links: LayoutLink[], drawn: LayoutLink[], gaps: LayoutGaps): LayoutResult {
  const { above, below } = segments(layers, links);
  for (const layer of layers) pack(layer, gaps.node, layer.map(() => null));
  for (let sweep = 0; sweep < SWEEPS; sweep += 1) {
    const downward = sweep % 2 === 0;
    const range = downward ? [...layers.keys()] : [...layers.keys()].reverse();
    const centres = new Map<string, number>();
    for (const layer of layers) for (const node of layer) centres.set(node.id, node.x + node.width / 2);
    for (const index of range) {
      const neighbours = downward ? above : below;
      const desired = layers[index].map((node) => {
        const value = median((neighbours.get(node.id) ?? []).map((id) => centres.get(id) as number));
        return value === null ? null : value - node.width / 2;
      });
      pack(layers[index], gaps.node, desired);
      for (const node of layers[index]) centres.set(node.id, node.x + node.width / 2);
    }
  }
  const placements = new Map<string, Placement>();
  const centres = new Map<string, number>();
  // Half a gap of headroom top and bottom, so a wire running back to the first rank has a
  // corridor to arrive through rather than cutting up through the boxes.
  const corridors = [0];
  let y = gaps.rank / 2;
  let width = 0;
  const positions = layers.flatMap((layer) => layer.map((node) => node.x));
  const left = positions.length === 0 ? 0 : Math.min(...positions);
  for (const layer of layers) {
    const height = layer.length === 0 ? 0 : Math.max(...layer.map((node) => node.height));
    for (const node of layer) {
      const x = node.x - left;
      if (node.lane) centres.set(node.id, x + LANE / 2);
      else {
        placements.set(node.id, { x, y: y + (height - node.height) / 2 });
        centres.set(node.id, x + node.width / 2);
      }
      width = Math.max(width, x + node.width);
    }
    y += height + gaps.rank / 2;
    corridors.push(y);
    y += gaps.rank / 2;
  }
  const routing = routesOf(layers, drawn, centres, corridors, width, gaps);
  return { placements, routes: routing.routes, width: routing.width, height: y - gaps.rank / 2 };
}

/**
 * Turns each link into the sequence of gaps it crosses. The wire drops through its own column to
 * the gap below its rank, steps sideways there into the next column, and repeats, so no sideways
 * run ever happens at a height where boxes live.
 */
function routesOf(
  layers: LayerNode[][],
  links: LayoutLink[],
  centres: Map<string, number>,
  corridors: number[],
  width: number,
  gaps: LayoutGaps,
): { routes: Map<string, Placement[]>; width: number } {
  const rankOf = new Map<string, number>();
  for (const [rank, layer] of layers.entries()) for (const node of layer) rankOf.set(node.id, rank);
  const routes = new Map<string, Placement[]>();
  let margin = width;
  for (const link of links) {
    const from = rankOf.get(link.source) as number;
    const to = rankOf.get(link.target) as number;
    const target = centres.get(link.target) as number;
    if (to > from) {
      const points: Placement[] = [];
      for (let rank = from; rank < to; rank += 1) {
        const next = rank + 1 === to ? link.target : dummyId(link, rank + 1);
        points.push({ x: centres.get(next) as number, y: corridors[rank + 1] });
      }
      routes.set(`${link.source}->${link.target}`, points);
      continue;
    }
    // A link that runs back up the graph — a residual or a loop — is taken around the side
    // rather than straight through the ranks it would otherwise cut across.
    margin += gaps.node;
    routes.set(`${link.source}->${link.target}`, [
      { x: margin, y: corridors[from + 1] },
      { x: margin, y: corridors[to] },
      { x: target, y: corridors[to] },
    ]);
  }
  return { routes, width: margin === width ? width : margin + gaps.node };
}

/** Keeps the layer order, honours the gap, and stays as close to the wanted positions as it can. */
function pack(layer: LayerNode[], gap: number, desired: (number | null)[]) {
  let cursor = -Infinity;
  for (const [index, node] of layer.entries()) {
    const wanted = desired[index] ?? node.x;
    node.x = Math.max(wanted, cursor);
    cursor = node.x + node.width + gap;
  }
  for (let index = layer.length - 2; index >= 0; index -= 1) {
    const wanted = desired[index];
    if (wanted === null || wanted <= layer[index].x) continue;
    layer[index].x = Math.min(wanted, layer[index + 1].x - layer[index].width - gap);
  }
}
