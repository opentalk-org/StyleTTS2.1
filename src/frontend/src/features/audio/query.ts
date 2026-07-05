import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { type AudioQuery, fetchAudioFiles } from "./api";

const KEY = "audio-files";

export function useAudioFilesQuery(params: AudioQuery) {
  return useQuery({
    queryKey: [KEY, params],
    queryFn: () => fetchAudioFiles(params),
    placeholderData: keepPreviousData,
  });
}
