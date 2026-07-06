import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { createRunner, fetchRunners } from "./api";

const RUNNER_REFRESH_MS = 15_000;

export function useRunnersQuery() {
  return useQuery({
    queryKey: ["runners"],
    queryFn: fetchRunners,
    staleTime: RUNNER_REFRESH_MS,
    refetchInterval: RUNNER_REFRESH_MS,
    refetchOnWindowFocus: false,
  });
}

export function useCreateRunnerMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createRunner,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runners"] }),
  });
}
