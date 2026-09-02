import { useQuery } from "@tanstack/react-query";

import { getArrayMetric, getArrayMetricNames, getModelGraph } from "@/data/api";


export function useModelGraph(runId: string, running: boolean) {
  return useQuery({
    queryKey: ["model-graph", runId],
    queryFn: () => getModelGraph(runId),
    staleTime: running ? 5_000 : Infinity,
    refetchInterval: running ? 5_000 : false,
    retry: running ? 3 : false,
  });
}

export function useArrayMetricNames(runId: string, running: boolean) {
  return useQuery({
    queryKey: ["array-metric-names", runId],
    queryFn: () => getArrayMetricNames(runId),
    refetchInterval: running ? 5_000 : false,
    retry: false,
  });
}

export function useArrayMetric(runId: string, name: string, running: boolean) {
  return useQuery({
    queryKey: ["array-metric", runId, name],
    queryFn: () => getArrayMetric(runId, name),
    refetchInterval: running ? 5_000 : false,
    retry: false,
  });
}
