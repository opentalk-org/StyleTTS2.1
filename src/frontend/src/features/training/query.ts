import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { createTrainingConfig, fetchTrainingConfigs, type CreateTrainingConfigPayload } from "./api";

const KEY = "training-configs";

export function useTrainingConfigsQuery(type: string) {
  return useQuery({
    queryKey: [KEY, type],
    queryFn: () => fetchTrainingConfigs(type),
  });
}

export function useCreateTrainingConfigMutation(type: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CreateTrainingConfigPayload) => createTrainingConfig(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [KEY, type] }),
  });
}
