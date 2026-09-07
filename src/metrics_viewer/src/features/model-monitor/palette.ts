import type { ModelTree, TreeNode } from "./hierarchy";

const HUES = 12;

/**
 * A hue per module, taken from the run palette. Containers step to a new hue, and everything drawn
 * inside one keeps its hue, so a box's colour says which block it belongs to at a glance. A child
 * never repeats its parent's hue and siblings never share one.
 */
export function hueIndex(tree: ModelTree): Map<string, number> {
  const hues = new Map<string, number>();
  const walk = (ids: string[], parent: number) => {
    for (const [position, id] of ids.entries()) {
      const node = tree.nodes.get(id) as TreeNode;
      const hue = node.childIds.length === 0 ? parent : (parent + 1 + position) % HUES;
      hues.set(id, hue < 0 ? 0 : hue);
      walk(node.childIds, hues.get(id) as number);
    }
  };
  walk(tree.rootIds, -1);
  return hues;
}

export function hueColor(hue: number): string {
  return `var(--series-${hue + 1})`;
}

/** The body of a box: its hue laid over the surface, dark on a dark theme and pale on a light one. */
export function hueSurface(hue: number): string {
  return `color-mix(in srgb, ${hueColor(hue)} 13%, var(--color-surface))`;
}

export function hueLine(hue: number, weight: number): string {
  return `color-mix(in srgb, ${hueColor(hue)} ${weight}%, var(--color-line))`;
}
