import { useQuery } from "@tanstack/react-query";

import type { Run } from "@/shared/types";

import { getArtifacts, runPlotsQuery } from "./server";

export function useArtifactsQuery(runs: Run[]) {
  const runIds = runs.map((run) => run.id);
  return useQuery({
    queryKey: ["artifacts", runIds],
    queryFn: () => getArtifacts({ data: runIds }),
    enabled: runIds.length > 0,
    staleTime: 30 * 60 * 1000,
  });
}

export function usePlotsQuery(projectId: string | null, sql: string, selectedRunIds: string[]) {
  const runIds = [...selectedRunIds].sort();
  return useQuery({
    queryKey: ["plots", projectId, sql, runIds],
    queryFn: () => runPlotsQuery({
      data: { sql, projectId: projectId as string, runIds: selectedRunIds },
    }),
    enabled: projectId !== null && selectedRunIds.length > 0,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
}
