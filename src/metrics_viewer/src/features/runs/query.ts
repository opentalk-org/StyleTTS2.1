import { useQuery } from "@tanstack/react-query";

import { listRuns } from "@/data/api";

export function useRunsQuery(projectId: string | null) {
  return useQuery({
    queryKey: ["runs", projectId],
    queryFn: () => listRuns(projectId as string),
    enabled: projectId !== null,
  });
}
