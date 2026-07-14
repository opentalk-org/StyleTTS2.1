import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { fetchWorkflowSchema } from "@/features/workflows/api";
import { showToast } from "@/shared/feedback/Toast";
import { type Checkpoint, deleteCheckpoint, fetchCheckpoints, renameCheckpoint, startCatalogDownload } from "./api";
import type { CatalogItem } from "./catalog";

const KEY = "checkpoints";

export function useCheckpointsQuery() {
  return useQuery({ queryKey: [KEY], queryFn: fetchCheckpoints });
}

export function useCheckpointActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: [KEY] });
  const rename = useMutation({
    mutationFn: ({ checkpoint, name }: { checkpoint: Checkpoint; name: string }) => renameCheckpoint(checkpoint, name),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: deleteCheckpoint,
    onSuccess: () => {
      showToast("Checkpoint deleted", undefined, "error");
      invalidate();
    },
  });
  return {
    rename: (checkpoint: Checkpoint, name: string) => rename.mutate({ checkpoint, name }),
    remove: (id: string) => remove.mutate(id),
  };
}

export function useCatalogDownloadMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (item: CatalogItem) => {
      const schema = await queryClient.fetchQuery({ queryKey: ["workflow-schema"], queryFn: fetchWorkflowSchema });
      return startCatalogDownload(item, schema);
    },
    onSuccess: (run, item) => {
      showToast(`Queued ${item.name}`, run.run_id);
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
    onError: (error, item) => {
      showToast(`Couldn't queue ${item.name}`, error instanceof Error ? error.message : undefined, "error");
    },
  });
}
