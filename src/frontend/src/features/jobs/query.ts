import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { type JobQuery, fetchJobs } from "./api";

export function useJobsQuery(params: JobQuery) {
  return useQuery({
    queryKey: ["jobs", params],
    queryFn: () => fetchJobs(params),
    placeholderData: keepPreviousData,
  });
}
