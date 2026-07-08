import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { deleteArtifact, fetchArtifacts } from "./api";

const KEY = "artifacts";

export function useArtifactsQuery() {
  return useQuery({ queryKey: [KEY], queryFn: fetchArtifacts });
}

export function useArtifactActions() {
  const qc = useQueryClient();
  const remove = useMutation({
    mutationFn: deleteArtifact,
    onSuccess: () => {
      showToast("Artifact deleted", undefined, "error");
      qc.invalidateQueries({ queryKey: [KEY] });
    },
  });
  return { remove: (id: string) => remove.mutate(id) };
}
