import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { fetchStorageSettings, updateStorageSettings, type StorageSettingsPayload } from "./api";

export function useStorageSettingsQuery() {
  return useQuery({ queryKey: ["storage-settings"], queryFn: fetchStorageSettings });
}

export function useStorageSettingsActions() {
  const queryClient = useQueryClient();
  const update = useMutation({
    mutationFn: updateStorageSettings,
    onSuccess: () => {
      showToast("Storage settings saved");
      queryClient.invalidateQueries({ queryKey: ["storage-settings"] });
    },
  });
  return { update: (payload: StorageSettingsPayload) => update.mutate(payload) };
}
