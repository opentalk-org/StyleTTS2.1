import { useQuery } from "@tanstack/react-query";

import { listRuns } from "./server";

export function useRunsQuery(projectId: string | null) {
  return useQuery({
    queryKey: ["runs", projectId],
    queryFn: () => listRuns({ data: projectId as string }),
    enabled: projectId !== null,
  });
}
