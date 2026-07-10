import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { AUDIO_FILES_KEY } from "@/features/audio/query";
import { fetchMosPair, saveMosRating } from "./api";

export const MOS_PAIR_KEY = "mos-pair";

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
      ]);
    },
  });
}
