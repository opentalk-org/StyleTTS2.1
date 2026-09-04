/** Applies a user order to items; unlisted items keep their relative order after listed ones. */
export function orderByList<T extends { name: string }>(items: T[], order: string[]): T[] {
  const rank = new Map(order.map((name, index) => [name, index]));
  return [...items].sort((a, b) => {
    const left = rank.get(a.name);
    const right = rank.get(b.name);
    if (left === undefined && right === undefined) return 0;
    if (left === undefined) return 1;
    if (right === undefined) return -1;
    return left - right;
  });
}

/**
 * Moves `from` onto `to`: before it when dragging backwards, after it when dragging
 * forwards, which is what a drop on a neighbouring item is expected to do.
 */
export function moveBefore(names: string[], from: string, to: string): string[] {
  if (from === to) return names;
  const toIndex = names.indexOf(to);
  const without = names.filter((name) => name !== from);
  return [...without.slice(0, toIndex), from, ...without.slice(toIndex)];
}
