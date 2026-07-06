import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { createTextFileAsset, fetchFileAssets, type CreateTextFileAssetPayload } from "./api";

const KEY = "file-assets";

export function useFileAssetsQuery(type?: string) {
  return useQuery({
    queryKey: [KEY, type ?? "all"],
    queryFn: () => fetchFileAssets(type),
  });
}

export function useCreateTextFileAssetMutation(type?: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CreateTextFileAssetPayload) => createTextFileAsset(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [KEY, type ?? "all"] });
      queryClient.invalidateQueries({ queryKey: [KEY, "all"] });
    },
  });
}
