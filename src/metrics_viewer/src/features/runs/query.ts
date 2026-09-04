import { useQuery } from "@tanstack/react-query";

import { getProjectBootstrap, getRunDetails, getRunMetrics } from "./server";

export function useProjectBootstrapQuery(projectId: string | null) {
  return useQuery({
    queryKey: ["project-bootstrap", projectId],
    queryFn: () => getProjectBootstrap({ data: projectId as string }),
    enabled: projectId !== null,
  });
}

export function useRunDetailsQuery(runIds: string[]) {
  const ids = [...runIds].sort();
  return useQuery({
    queryKey: ["run-details", ids],
    queryFn: () => getRunDetails({ data: ids }),
    enabled: ids.length > 0,
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  });
}

export function useRunMetricsQuery(projectId: string | null, names: string[]) {
  const metricNames = [...names].sort();
  return useQuery({
    queryKey: ["run-metrics", projectId, metricNames],
    queryFn: () => getRunMetrics({ data: { projectId: projectId as string, names: metricNames } }),
    enabled: projectId !== null && metricNames.length > 0,
    placeholderData: (previous) => previous,
    staleTime: Infinity,
  });
}
