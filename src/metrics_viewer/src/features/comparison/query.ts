import { useQuery } from "@tanstack/react-query";

import { getArtifacts, runPlotsQuery } from "@/data/api";
import type { Run } from "@/shared/types";

export function useArtifactsQuery(runs: Run[]) {
  const runIds = runs.map((run) => run.id);
  return useQuery({
    queryKey: ["artifacts", runIds],
    queryFn: () => getArtifacts(runIds),
    enabled: runIds.length > 0,
    staleTime: 30 * 60 * 1000,
  });
}







export function usePlotsQuery(projectId: string | null, sql: string, selectedRunIds: string[]) {
  const runIds = [...selectedRunIds].sort();
  return useQuery({
    queryKey: ["plots", projectId, sql, runIds],
    queryFn: () => runPlotsQuery(sql, projectId as string, selectedRunIds),
    enabled: projectId !== null && selectedRunIds.length > 0,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
}
