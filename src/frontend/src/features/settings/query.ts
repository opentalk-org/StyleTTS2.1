import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  fetchIntegrationSettings,
  fetchStorageSettings,
  testStorageSettings,
  updateIntegrationSettings,
  updateStorageSettings,
} from "./api";

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

export function useStorageConnectionTest() {
  return useMutation({ mutationFn: testStorageSettings });
}

export function useIntegrationSettingsQuery() {
  return useQuery({ queryKey: ["integration-settings"], queryFn: fetchIntegrationSettings });
}

export function useIntegrationSettingsActions() {
  const queryClient = useQueryClient();
  const update = useMutation({
    mutationFn: updateIntegrationSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integration-settings"] });
    },
  });
  return update;
}
