import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { fetchStorageSettings, updateStorageSettings } from "./api";

export function useStorageSettingsQuery() {
  return useQuery({ queryKey: ["storage-settings"], queryFn: fetchStorageSettings });
}

export function useStorageSettingsActions() {
  const queryClient = useQueryClient();
  const update = useMutation({
    mutationFn: updateStorageSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["storage-settings"] });
    },
  });
  return update;
}
