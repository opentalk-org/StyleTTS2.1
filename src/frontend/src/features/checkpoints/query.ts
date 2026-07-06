import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { type Checkpoint, deleteCheckpoint, fetchCheckpoints, renameCheckpoint } from "./api";

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
