import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { createDataset, deleteDataset, fetchDatasets } from "./api";

export const DATASETS_KEY = "datasets";

export function useDatasetsQuery() {
  return useQuery({
    queryKey: [DATASETS_KEY],
    queryFn: fetchDatasets,
  });
}

export function useDatasetActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: [DATASETS_KEY] });

  const create = useMutation({
    mutationFn: createDataset,
    onSuccess: (dataset) => {
      showToast(`Dataset "${dataset.name}" created`);
      invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => {
      showToast("Dataset deleted", undefined, "error");
      invalidate();
    },
  });

  return {
    create: (name: string) => create.mutate(name),
    remove: (id: string) => remove.mutate(id),
  };
}
