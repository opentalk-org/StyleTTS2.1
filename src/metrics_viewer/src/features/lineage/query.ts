import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { useViewerStore } from "@/features/viewer/store";
import type { Lineage } from "@/shared/types";

import { ancestorCuts, lineageGroups, lineageOffsets, runChains, withAncestors, type RunChain } from "./chains";
import { getLineage } from "./server";

const EMPTY: Lineage = { checkpoints: [], runs: [], firstSteps: {} };

export function useLineageQuery(projectId: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ["lineage", projectId],
    queryFn: () => getLineage({ data: projectId as string }),
    enabled: enabled && projectId !== null,
    staleTime: 60_000,
  });
}

export interface RunLineage {
  chains: Map<string, RunChain>;
  /** Run id to the step its own step 0 sits at on the lineage axis. */
  offsets: Map<string, number>;
  /** Names of runs the lineage knows about, for ancestors missing from the runs list. */
  names: Map<string, string>;
  /** No checkpoint of this project records a resume, so there is nothing to follow. */
  empty: boolean;
}

/**
 * The run-level resume graph of the open project. Shared by the lineage panel, the run
 * scope and the lineage x axis; the underlying query is cached, so calling it from several
 * components costs one request.
 */
export function useRunLineage(): RunLineage {
  const projectId = useViewerStore((state) => state.projectId);
  const { data } = useLineageQuery(projectId, true);
  return useMemo(() => {
    const lineage = data ?? EMPTY;
    const chains = runChains(lineage);
    return {
      chains,
      offsets: lineageOffsets(chains),
      names: new Map(lineage.runs.map((run) => [run.id, run.name])),
      empty: ![...chains.values()].some((chain) => chain.parentRunId !== null),
    };
  }, [data]);
}

const NO_ANCESTORS: ReadonlySet<string> = new Set();

/**
 * Runs the charts draw only because a selected run was resumed from them — empty unless
 * the lineage scope is on. They are drawn dashed and faded so the selection stays readable.
 */
export function useAncestorRunIds(): ReadonlySet<string> {
  const selectedRunIds = useViewerStore((state) => state.selectedRunIds);
  const runScope = useViewerStore((state) => state.runScope);
  const { chains } = useRunLineage();
  return useMemo(() => {
    if (runScope !== "lineage") return NO_ANCESTORS;
    return new Set(withAncestors(selectedRunIds, chains).filter((id) => !selectedRunIds.includes(id)));
  }, [chains, runScope, selectedRunIds]);
}

const NO_GROUPS: Map<string, ReadonlySet<string>> = new Map();

/**
 * Run id to every run drawn from the same lineage. Only populated in the ancestor scope:
 * with just the selected runs on screen there is no chain to highlight together.
 */
export function useLineageGroups(runIds: string[]): Map<string, ReadonlySet<string>> {
  const runScope = useViewerStore((state) => state.runScope);
  const { chains } = useRunLineage();
  const key = runIds.join(",");
  return useMemo(
    () => (runScope === "lineage" ? lineageGroups(key === "" ? [] : key.split(","), chains) : NO_GROUPS),
    [chains, key, runScope],
  );
}

const NO_CUTS: Map<string, number> = new Map();

/**
 * Last step to draw for each run that is only on the chart as an ancestor. Empty outside
 * the ancestor scope, where every run drawn was asked for explicitly.
 */
export function useAncestorCuts(runIds: string[]): Map<string, number> {
  const selectedRunIds = useViewerStore((state) => state.selectedRunIds);
  const runScope = useViewerStore((state) => state.runScope);
  const { chains } = useRunLineage();
  const key = runIds.join(",");
  return useMemo(
    () => (runScope === "lineage" ? ancestorCuts(key === "" ? [] : key.split(","), selectedRunIds, chains) : NO_CUTS),
    [chains, key, runScope, selectedRunIds],
  );
}
