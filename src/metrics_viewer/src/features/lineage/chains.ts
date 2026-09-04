import type { Checkpoint, Lineage } from "@/shared/types";

export interface RunChain {
  runId: string;
  /** Run this one resumed from, or null when it started from scratch. */
  parentRunId: string | null;
  /** Checkpoint it resumed from — the fork point drawn in the graph. */
  forkCheckpointId: string | null;
  /** Step the parent had reached at that checkpoint. */
  resumeStep: number;
  /**
   * Shift applied to this run's steps to place it on the lineage axis. A run that restarted
   * its step counter is pushed out to where it was resumed; one that carried its parent's
   * counter on is already in the right place and only inherits its parent's shift.
   */
  offset: number;
  /** True when the run's own steps start again from the beginning rather than continuing. */
  restarted: boolean;
}

/**
 * Run-level view of the checkpoint graph. A run resumes from another run at the checkpoint
 * whose ancestor belongs to a different run; anything else — no ancestor, or an ancestor
 * whose asset is outside this project — reads as a root.
 */
export function runChains(lineage: Lineage): Map<string, RunChain> {
  const byId = new Map(lineage.checkpoints.map((checkpoint) => [checkpoint.id, checkpoint]));
  const resumeOf = new Map<string, Checkpoint>();
  for (const checkpoint of lineage.checkpoints) {
    const ancestor = checkpoint.ancestorId === null ? undefined : byId.get(checkpoint.ancestorId);
    if (ancestor === undefined || ancestor.runId === checkpoint.runId) continue;
    const known = resumeOf.get(checkpoint.runId);
    if (known === undefined || checkpoint.createdAt < known.createdAt) resumeOf.set(checkpoint.runId, ancestor);
  }

  const chains = new Map<string, RunChain>();
  for (const runId of new Set(lineage.checkpoints.map((checkpoint) => checkpoint.runId))) {
    const ancestor = resumeOf.get(runId);
    const resumeStep = ancestor?.step ?? 0;
    chains.set(runId, {
      runId,
      parentRunId: ancestor?.runId ?? null,
      forkCheckpointId: ancestor?.id ?? null,
      resumeStep,
      offset: 0,
      // A run whose first step is at or below the step it was resumed at started counting
      // over; one that logs past it is carrying the parent's count on and needs no shift.
      restarted: (lineage.firstSteps[runId] ?? 0) <= resumeStep,
    });
  }

  // Offsets accumulate down the chain; a cycle stops at the run that closes it.
  const resolve = (runId: string, seen: Set<string>): number => {
    const chain = chains.get(runId);
    if (chain === undefined || chain.parentRunId === null || seen.has(runId)) return 0;
    const parent = resolve(chain.parentRunId, seen.add(runId));
    return chain.restarted ? parent + chain.resumeStep : parent;
  };
  for (const chain of chains.values()) chain.offset = resolve(chain.runId, new Set());
  return chains;
}

/** Every run `runId` was resumed from, nearest first. */
export function ancestorRunIds(runId: string, chains: Map<string, RunChain>): string[] {
  const result: string[] = [];
  let current = chains.get(runId)?.parentRunId ?? null;
  while (current !== null && !result.includes(current)) {
    result.push(current);
    current = chains.get(current)?.parentRunId ?? null;
  }
  return result;
}

/** The runs themselves plus everything they were resumed from, selection order kept first. */
export function withAncestors(runIds: string[], chains: Map<string, RunChain>): string[] {
  const seen = new Set(runIds);
  const extra: string[] = [];
  for (const runId of runIds) {
    for (const ancestorId of ancestorRunIds(runId, chains)) {
      if (seen.has(ancestorId)) continue;
      seen.add(ancestorId);
      extra.push(ancestorId);
    }
  }
  return [...runIds, ...extra];
}

/** Lineage x of a run's step: its own step shifted by where the run starts in the chain. */
export function lineageOffsets(chains: Map<string, RunChain>): Map<string, number> {
  return new Map([...chains.values()].map((chain) => [chain.runId, chain.offset]));
}

/**
 * Groups the given runs into the lineages they belong to: a run, everything it resumed
 * from and everything resumed from it, as far as the drawn set reaches. Hovering one curve
 * can then highlight the whole chain instead of a single run of it.
 */
export function lineageGroups(runIds: string[], chains: Map<string, RunChain>): Map<string, ReadonlySet<string>> {
  const drawn = new Set(runIds);
  const parent = new Map(runIds.map((runId) => [runId, runId]));
  const find = (runId: string): string => {
    let root = runId;
    while (parent.get(root) !== root) root = parent.get(root) as string;
    for (let node = runId; node !== root; ) {
      const next = parent.get(node) as string;
      parent.set(node, root);
      node = next;
    }
    return root;
  };
  for (const runId of runIds) {
    const parentRunId = chains.get(runId)?.parentRunId ?? null;
    if (parentRunId === null || !drawn.has(parentRunId)) continue;
    parent.set(find(runId), find(parentRunId));
  }

  const members = new Map<string, Set<string>>();
  for (const runId of runIds) {
    const root = find(runId);
    const group = members.get(root) ?? new Set<string>();
    group.add(runId);
    members.set(root, group);
  }
  return new Map(runIds.map((runId) => [runId, members.get(find(runId)) as ReadonlySet<string>]));
}

/**
 * Where each ancestor's history stops being history of the selection.
 *
 * A run forked from the middle of another one: the parent kept training past that
 * checkpoint, and those later steps are not part of the child's past. So a run drawn only
 * because something resumed from it is cut at the last step any drawn child forked at —
 * which is also what makes the two curves meet exactly on the lineage axis instead of
 * overlapping. A run that is selected in its own right is never cut.
 */
export function ancestorCuts(
  drawnRunIds: string[],
  selectedRunIds: string[],
  chains: Map<string, RunChain>,
): Map<string, number> {
  const drawn = new Set(drawnRunIds);
  const cuts = new Map<string, number>();
  for (const runId of drawnRunIds) {
    const chain = chains.get(runId);
    if (chain === undefined || chain.parentRunId === null || !drawn.has(chain.parentRunId)) continue;
    const known = cuts.get(chain.parentRunId);
    if (known === undefined || chain.resumeStep > known) cuts.set(chain.parentRunId, chain.resumeStep);
  }
  for (const runId of selectedRunIds) cuts.delete(runId);
  return cuts;
}
