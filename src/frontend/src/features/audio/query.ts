import { useEffect, useState } from "react";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { type AudioDeleteRequest, type AudioQuery, deleteAudioFiles, deleteMatchingAudioFiles, fetchAudioFile, fetchAudioFiles, fetchWaveform } from "./api";

export const AUDIO_FILES_KEY = "audio-files";
const WAVEFORM_DEBOUNCE_MS = 200;
const WAVEFORM_TIME_STEP = 0.25;

type WaveformWindow = {
  id: string | null;
  start: number;
  end: number;
  points: number;
};

export function useAudioFilesQuery(params: AudioQuery) {
  return useQuery({
    queryKey: [AUDIO_FILES_KEY, params],
    queryFn: () => fetchAudioFiles(params),
    placeholderData: keepPreviousData,
  });
}

export function useAudioFileQuery(id: string | null) {
  return useQuery({
    queryKey: [AUDIO_FILES_KEY, id],
    queryFn: () => fetchAudioFile(id!),
    enabled: id !== null,
  });
}

export function useDeleteAudioFilesMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: AudioDeleteRequest) => {
      if (request.mode === "filter") return deleteMatchingAudioFiles(request.query, request.dataset);
      return deleteAudioFiles(request.ids);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] }),
  });
}

export function useWaveformQuery(id: string | null, start: number, end: number, points: number) {
  const [target, setTarget] = useState<WaveformWindow>(() => waveformWindow(id, start, end, points));

  useEffect(() => {
    const timeout = globalThis.setTimeout(() => {
      setTarget(waveformWindow(id, start, end, points));
    }, WAVEFORM_DEBOUNCE_MS);
    return () => globalThis.clearTimeout(timeout);
  }, [id, start, end, points]);

  return useQuery({
    queryKey: [AUDIO_FILES_KEY, "waveform", target.id, target.start, target.end, target.points],
    queryFn: () => fetchWaveform(target.id!, target.start, target.end, target.points),
    enabled: target.id === id && target.id !== null && target.end > target.start,
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
}

function waveformWindow(id: string | null, start: number, end: number, points: number): WaveformWindow {
  const safeStart = floorWaveformTime(Math.max(0, start));
  const safeEnd = ceilWaveformTime(Math.max(safeStart, end));
  return { id, start: safeStart, end: safeEnd, points };
}

function floorWaveformTime(value: number): number {
  return Number((Math.floor(value / WAVEFORM_TIME_STEP) * WAVEFORM_TIME_STEP).toFixed(3));
}

function ceilWaveformTime(value: number): number {
  return Number((Math.ceil(value / WAVEFORM_TIME_STEP) * WAVEFORM_TIME_STEP).toFixed(3));
}
