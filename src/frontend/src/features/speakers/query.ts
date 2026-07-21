import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import {
  type SpeakerDeleteRequest,
  type SpeakerQuery,
  deleteMatchingSpeakers,
  deleteSpeaker,
  deleteSpeakers,
  fetchSpeakers,
  renameSpeaker,
} from "./api";

const KEY = "speakers";

/**
 * Query a filtered page of speakers. Filters are part of the query key so the
 * server (mock) does the work; `keepPreviousData` avoids a loading flash while
 * typing in the search box.
 */
export function useSpeakersQuery(params: SpeakerQuery) {
  return useQuery({
    queryKey: [KEY, params],
    queryFn: () => fetchSpeakers(params),
    placeholderData: keepPreviousData,
  });
}

export function useSpeakerActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: [KEY] });

  const rename = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameSpeaker(id, name),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: deleteSpeaker,
    onSuccess: () => {
      showToast("Speaker deleted", undefined, "error");
      invalidate();
    },
  });

  return {
    rename: (id: string, name: string) => rename.mutate({ id, name }),
    remove: (id: string) => remove.mutate(id),
  };
}

export function useDeleteSpeakersMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (request: SpeakerDeleteRequest) => {
      if (request.mode === "filter") return deleteMatchingSpeakers(request.query);
      return deleteSpeakers(request.ids);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: [KEY] }),
  });
}
