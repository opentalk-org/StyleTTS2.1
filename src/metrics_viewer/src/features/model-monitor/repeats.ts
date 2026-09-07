import type { ModelTree, TreeNode } from "./hierarchy";

/** A set of siblings drawn as one box, labelled with how many of them it stands for. */
export interface RepeatGroup {
  /** Id of the first member, which stands in for the whole set. */
  id: string;
  memberIds: string[];
}

export interface RepeatIndex {
  /** Member id to the id of the group it belongs to. */
  groupOf: Map<string, string>;
  groups: Map<string, RepeatGroup>;
}

/**
 * Structural fingerprint of a subtree: module types, parameter shapes and counts, and child names,
 * but no ids or step numbers, so `encoder.0` and `encoder.7` of one stack fingerprint alike while
 * two stages of differing width do not.
 */
function signatures(tree: ModelTree): Map<string, string> {
  const cache = new Map<string, string>();
  const of = (id: string): string => {
    const cached = cache.get(id);
    if (cached !== undefined) return cached;
    const node = tree.nodes.get(id) as TreeNode;
    const shapes = Object.entries(node.component?.parameter_shapes ?? {})
      .map(([name, shape]) => `${name}:${shape}`)
      .join(",");
    const children = node.childIds.map((childId) => `${(tree.nodes.get(childId) as TreeNode).name}=${of(childId)}`);
    const value = `${node.moduleType}(${shapes};${node.parameterCount})[${children.join("|")}]`;
    cache.set(id, value);
    return value;
  };
  for (const id of tree.nodes.keys()) of(id);
  return cache;
}

/**
 * Folds each container's children twice: first every extra invocation of one module onto its first
 * invocation, which turns an unrolled loop back into a single step, then every run of identical
 * consecutive siblings onto its head, which turns a stack of blocks into one.
 */
export function findRepeats(tree: ModelTree): RepeatIndex {
  const signature = signatures(tree);
  const groupOf = new Map<string, string>();
  const groups = new Map<string, RepeatGroup>();
  const fold = (childIds: string[]) => {
    foldRuns(tree, signature, foldInvocations(tree, childIds, groupOf), groupOf);
    for (const childId of childIds) {
      const head = groupOf.get(childId);
      if (head === undefined) continue;
      const group = groups.get(head);
      if (group === undefined) groups.set(head, { id: head, memberIds: [head, childId] });
      else group.memberIds.push(childId);
    }
    for (const childId of childIds) fold((tree.nodes.get(childId) as TreeNode).childIds);
  };
  fold(tree.rootIds);
  return { groupOf, groups };
}

/** Later invocations of a module carry the same path, so they join the box of the first one. */
function foldInvocations(tree: ModelTree, childIds: string[], groupOf: Map<string, string>): string[] {
  const firstOfPath = new Map<string, string>();
  const heads: string[] = [];
  for (const childId of childIds) {
    const path = (tree.nodes.get(childId) as TreeNode).modulePath;
    const first = firstOfPath.get(path);
    if (path.length > 0 && first !== undefined) groupOf.set(childId, first);
    else {
      firstOfPath.set(path, childId);
      heads.push(childId);
    }
  }
  return heads;
}

/**
 * Runs of identical consecutive siblings collapse onto their head, taking with them anything
 * already folded onto a member of the run.
 */
function foldRuns(tree: ModelTree, signature: Map<string, string>, heads: string[], groupOf: Map<string, string>) {
  let start = 0;
  while (start < heads.length) {
    let end = start + 1;
    while (end < heads.length && continues(tree, signature, heads[start], heads[end], end - start)) end += 1;
    for (const memberId of heads.slice(start + 1, end)) {
      for (const [foldedId, head] of groupOf) if (head === memberId) groupOf.set(foldedId, heads[start]);
      groupOf.set(memberId, heads[start]);
    }
    start = end;
  }
}

/**
 * A member continues a run when it has the same fingerprint and either the next index of the same
 * stack, or the same name — module children are uniquely named, so equal names mean traced
 * operations, whose repetitions are worth folding just as much as a block stack.
 */
function continues(tree: ModelTree, signature: Map<string, string>, firstId: string, candidateId: string, offset: number): boolean {
  const first = tree.nodes.get(firstId) as TreeNode;
  const candidate = tree.nodes.get(candidateId) as TreeNode;
  if (signature.get(firstId) !== signature.get(candidateId)) return false;
  const index = Number(first.name);
  if (Number.isInteger(index)) return Number(candidate.name) === index + offset;
  return candidate.name === first.name;
}
