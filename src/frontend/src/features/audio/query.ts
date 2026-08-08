import { useEffect, useState } from "react";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  type AddToDatasetRequest,
  type AudioDeleteRequest,
  type AudioQuery,
  addAudioFilesToDataset,
  deleteAudioFiles,
  deleteMatchingAudioFiles,
  ensureWaveform,
  fetchAudioFile,
  fetchAudioFiles,
  fetchAudioSegmentPreview,
  fetchWaveform,
  renameAudioFile,
} from "./api";
import { DATASETS_KEY } from "@/features/datasets/query";

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
  });
}

export function useAudioFileQuery(id: string | null) {
  return useQuery({
    queryKey: [AUDIO_FILES_KEY, id],
    queryFn: () => fetchAudioFile(id!),
    enabled: id !== null,
  });
}

export function useAudioSegmentPreviewQuery(id: string, enabled: boolean) {
  return useQuery({
    queryKey: [AUDIO_FILES_KEY, "segment-preview", id],
    queryFn: () => fetchAudioSegmentPreview(id),
    enabled,
    staleTime: 60_000,
  });
}

export function useDeleteAudioFilesMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: AudioDeleteRequest) => {
      if (request.mode === "filter") return deleteMatchingAudioFiles(request.query, request.language, request.dataset);
      return deleteAudioFiles(request.ids);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] }),
  });
}

export function useAddToDatasetMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: AddToDatasetRequest) => addAudioFilesToDataset(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] });
      queryClient.invalidateQueries({ queryKey: [DATASETS_KEY] });
    },
  });
}

export function useRenameAudioFileMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameAudioFile(id, name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] }),
  });
}

export function useWaveformStatusQuery(id: string | null) {
  return useQuery({
    queryKey: [AUDIO_FILES_KEY, "waveform-status", id],
    queryFn: () => ensureWaveform(id!),
    enabled: id !== null,
    refetchInterval: (query) => (query.state.data?.status === "pending" ? 1000 : false),
    refetchOnWindowFocus: false,
  });
}

export function useWaveformQuery(id: string | null, start: number, end: number, points: number, ready: boolean) {
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
    enabled: ready && target.id === id && target.id !== null && target.end > target.start,
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
