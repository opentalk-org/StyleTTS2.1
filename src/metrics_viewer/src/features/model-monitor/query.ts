import { useQuery } from "@tanstack/react-query";

import { getArrayMetric, getArrayMetricNames, getModelGraph } from "./server";

export function useModelGraph(runId: string, running: boolean) {
  return useQuery({
    queryKey: ["model-graph", runId],
    queryFn: () => getModelGraph({ data: runId }),
    staleTime: Infinity,
    retry: running ? 3 : false,
  });
}

export function useArrayMetricNames(runId: string, running: boolean) {
  return useQuery({
    queryKey: ["array-metric-names", runId],
    queryFn: () => getArrayMetricNames({ data: runId }),
    staleTime: Infinity,
    retry: false,
  });
}

export function useArrayMetric(runId: string, name: string, running: boolean) {
  return useQuery({
    queryKey: ["array-metric", runId, name],
    queryFn: () => getArrayMetric({ data: { runId, name } }),
    staleTime: Infinity,
    retry: false,
  });
}
