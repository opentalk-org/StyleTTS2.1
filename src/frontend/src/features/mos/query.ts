import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { AUDIO_FILES_KEY } from "@/features/audio/query";
import { fetchMosPair, fetchMosRatings, saveMosRating, undoMosRating, updateMosRating } from "./api";
import type { MosRatingUpdateRequest } from "./api";

export const MOS_PAIR_KEY = "mos-pair";
export const MOS_HISTORY_KEY = "mos-history";
const HISTORY_PAGE_SIZE = 100;

export function useMosPairQuery(datasetIds: string[]) {
  const orderedIds = [...datasetIds].sort();
  return useQuery({
    queryKey: [MOS_PAIR_KEY, orderedIds],
    queryFn: () => fetchMosPair(orderedIds),
    enabled: orderedIds.length > 0,
    refetchOnWindowFocus: false,
  });
}

export function useSaveMosRatingMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: saveMosRating,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] }),
        queryClient.invalidateQueries({ queryKey: [MOS_PAIR_KEY] }),
        queryClient.invalidateQueries({ queryKey: [MOS_HISTORY_KEY] }),
      ]);
    },
  });
}

export function useMosHistoryQuery(datasetIds: string[]) {
  const orderedIds = [...datasetIds].sort();
  return useInfiniteQuery({
    queryKey: [MOS_HISTORY_KEY, orderedIds],
    queryFn: ({ pageParam }) => fetchMosRatings(orderedIds, HISTORY_PAGE_SIZE, pageParam),
    initialPageParam: 0,
    getNextPageParam: (page) => page.offset + page.rows.length < page.total
      ? page.offset + page.rows.length
      : undefined,
    enabled: orderedIds.length > 0,
    refetchOnWindowFocus: false,
  });
}

export function useMosHistoryMutations() {
  const queryClient = useQueryClient();
  const invalidate = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] }),
      queryClient.invalidateQueries({ queryKey: [MOS_PAIR_KEY] }),
      queryClient.invalidateQueries({ queryKey: [MOS_HISTORY_KEY] }),
    ]);
  };
  const update = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: MosRatingUpdateRequest }) => updateMosRating(id, payload),
    onSuccess: invalidate,
  });
  const undo = useMutation({ mutationFn: undoMosRating, onSuccess: invalidate });
  return { update, undo };
}
