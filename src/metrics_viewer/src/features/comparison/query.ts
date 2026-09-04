import { useQuery } from "@tanstack/react-query";

import type { Run } from "@/shared/types";

import { getArtifacts, getPlotRange, runPlotsQuery } from "./server";

export function useArtifactsQuery(runs: Run[], enabled: boolean) {
  const runIds = runs.map((run) => run.id);
  return useQuery({
    queryKey: ["artifacts", runIds],
    queryFn: () => getArtifacts({ data: runIds }),
    enabled: enabled && runIds.length > 0,
    staleTime: 30 * 60 * 1000,
    placeholderData: (previous) => previous,
  });
}

export function usePlotsQuery(projectId: string | null, sql: string, selectedRunIds: string[], enabled = true) {
  const runIds = [...selectedRunIds].sort();
  return useQuery({
    queryKey: ["plots", projectId, sql, runIds],
    queryFn: () => runPlotsQuery({
      data: { sql, projectId: projectId as string, runIds: selectedRunIds },
    }),
    enabled: enabled && projectId !== null && selectedRunIds.length > 0,
    staleTime: 5 * 60 * 1000,
    placeholderData: (previous) => previous,
    retry: false,
  });
}

export interface PlotRange {
  xMin: number | null;
  xMax: number | null;
}

export function usePlotRangeQuery(
  runIds: string[],
  metric: string,
  range: PlotRange,
  targetPoints: number,
  enabled: boolean,
) {
  const ids = [...runIds].sort();
  return useQuery({
    queryKey: ["plot-range", ids, metric, range.xMin, range.xMax, targetPoints],
    queryFn: () => getPlotRange({ data: { runIds: ids, metric, ...range, targetPoints } }),
    enabled: enabled && ids.length > 0,
    staleTime: Infinity,
    placeholderData: (previous) => previous,
    retry: false,
  });
}
