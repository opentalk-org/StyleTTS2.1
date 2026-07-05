import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { createRunner, fetchRunners } from "./api";

export function useRunnersQuery() {
  return useQuery({ queryKey: ["runners"], queryFn: fetchRunners, refetchInterval: 3000 });
}

export function useCreateRunnerMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createRunner,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runners"] }),
  });
}
