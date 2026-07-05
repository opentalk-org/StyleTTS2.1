import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { type VoiceQuery, createVoice, deleteVoice, fetchVoices, renameVoice } from "./api";

const KEY = "voices";

/**
 * Query a filtered page of voices. Filters are part of the query key so the
 * server (mock) does the work; `keepPreviousData` avoids a loading flash while
 * typing in the search box.
 */
export function useVoicesQuery(params: VoiceQuery) {
  return useQuery({
    queryKey: [KEY, params],
    queryFn: () => fetchVoices(params),
    placeholderData: keepPreviousData,
  });
}

/** Mutations invalidate the voices cache so the filtered page refetches. */
export function useVoiceActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: [KEY] });

  const add = useMutation({
    mutationFn: createVoice,
    onSuccess: (name) => {
      showToast(`Voice "${name}" created`);
      invalidate();
    },
  });
  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameVoice(id, name),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: deleteVoice,
    onSuccess: () => {
      showToast("Voice deleted", undefined, "error");
      invalidate();
    },
  });

  return {
    add: () => add.mutate(),
    rename: (id: string, name: string) => rename.mutate({ id, name }),
    remove: (id: string) => remove.mutate(id),
  };
}
