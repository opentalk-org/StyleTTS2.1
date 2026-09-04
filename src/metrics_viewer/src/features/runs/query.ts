import { useQueries, useQuery } from "@tanstack/react-query";

import { getRunParams, getRunSummary, listRuns } from "./server";

export function useRunsQuery(projectId: string | null) {
  return useQuery({
    queryKey: ["runs", projectId],
    queryFn: () => listRuns({ data: projectId as string }),
    enabled: projectId !== null,
  });
}

export function useRunDetailsQueries(runIds: string[]) {
  const params = useQueries({
    queries: runIds.map((runId) => ({
      queryKey: ["run-params", runId],
      queryFn: () => getRunParams({ data: runId }),
      staleTime: Infinity,
    })),
  });
  const summaries = useQueries({
    queries: runIds.map((runId) => ({
      queryKey: ["run-summary", runId],
      queryFn: () => getRunSummary({ data: runId }),
      staleTime: Infinity,
    })),
  });
  return Object.fromEntries(runIds.map((runId, index) => [runId, {
    params: params[index].data ?? {},
    summary: summaries[index].data ?? {},
  }]));
}
