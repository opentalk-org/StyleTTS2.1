import { backendFetch, backendRequest, backendResourceUrl } from "@/app/backend";
import type { AudioSort } from "./store";

export type AudioSegment = {
  id: string;
  start: number;
  end: number;
  text: string;
  phon: string;
  speaker: string;
};

export type Segment = AudioSegment;

export type AudioFile = {
  id: string;
  name: string;
  speaker: string;
  duration: number;
  sample_rate: number | null;
  byte_length: number;
  size_mb: string;
  segments: number;
  segment_preview: AudioSegment[];
  dataset_ids: string[];
  virtual: boolean;
  metadata: Record<string, unknown>;
  updated_at: string;
};

export type AudioQuery = {
  query: string;
  dataset: string;
  sort: AudioSort;
  limit: number;
  offset: number;
};

export type AudioPage = {
  rows: AudioFile[];
  total: number;
};

export type AudioUploadOptions = {
  datasetId: string;
  speaker: string;
};

export type AudioUploadProgress = {
  fileName: string;
  fileIndex: number;
  fileCount: number;
  filePercent: number;
  overallPercent: number;
  phase: "decoding" | "uploading";
};

export type AudioDeleteRequest =
  | { mode: "ids"; ids: string[] }
  | { mode: "filter"; query: string; dataset: string };

export type WaveformRead = {
  duration: number;
  sample_rate: number;
  points_per_second: number;
  start: number;
  end: number;
  peaks: [number, number][];
};

export function fetchAudioFiles(params: AudioQuery): Promise<AudioPage> {
  const search = new URLSearchParams({
    query: params.query,
    dataset: params.dataset,
    sort: params.sort,
    limit: String(params.limit),
    offset: String(params.offset),
  });
  return backendRequest<AudioPage>(`/audio-files?${search}`);
}

export function fetchAudioFile(id: string): Promise<AudioFile> {
  return backendRequest<AudioFile>(`/audio-files/${encodeURIComponent(id)}`);
}

export function saveAudioSegments(id: string, segments: Segment[]): Promise<AudioFile> {
  return backendRequest<AudioFile>(`/audio-files/${encodeURIComponent(id)}/segments`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });
}

export async function deleteAudioFile(id: string): Promise<void> {
  const response = await backendFetch(`/audio-files/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export async function deleteAudioFiles(ids: string[]): Promise<void> {
  await Promise.all(ids.map((id) => deleteAudioFile(id)));
}

export async function deleteMatchingAudioFiles(query: string, dataset: string): Promise<void> {
  const search = new URLSearchParams({ query, dataset });
  const response = await backendFetch(`/audio-files?${search}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
}

export function fetchWaveform(id: string, start: number, end: number, points: number): Promise<WaveformRead> {
  const search = new URLSearchParams({ start: String(start), end: String(end), points: String(points) });
  return backendRequest<WaveformRead>(`/audio-files/${encodeURIComponent(id)}/waveform?${search}`);
}

export async function uploadAudioFiles(
  files: File[],
  options: AudioUploadOptions,
  onProgress?: (progress: AudioUploadProgress) => void,
): Promise<AudioFile[]> {
  const context = new AudioContext();
  try {
    const uploaded: AudioFile[] = [];
    for (const [index, file] of files.entries()) {
      onProgress?.(uploadProgress(file, index, files.length, 0, "decoding"));
      const decoded = await context.decodeAudioData(await file.arrayBuffer());
      uploaded.push(await uploadAudioFile(file, options, decoded.duration, decoded.sampleRate, waveformPayload(decoded), (filePercent) => {
        onProgress?.(uploadProgress(file, index, files.length, filePercent, "uploading"));
      }));
      onProgress?.(uploadProgress(file, index, files.length, 100, "uploading"));
    }
    return uploaded;
  } finally {
    await context.close();
  }
}

function uploadAudioFile(
  file: File,
  options: AudioUploadOptions,
  duration: number,
  sampleRate: number,
  waveform: string,
  onProgress: (filePercent: number) => void,
): Promise<AudioFile> {
  const body = new FormData();
  body.append("file", file);
  body.append("duration", String(duration));
  body.append("sample_rate", String(sampleRate));
  body.append("dataset_id", options.datasetId);
  body.append("speaker", options.speaker);
  body.append("waveform", waveform);
  return uploadForm<AudioFile>("/audio-files/upload", body, onProgress);
}

function uploadForm<T>(path: string, body: FormData, onProgress: (filePercent: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", backendResourceUrl(path));
    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onProgress(Math.round((event.loaded / event.total) * 100));
    };
    request.onload = () => {
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(`Backend request failed: ${request.status}`));
        return;
      }
      resolve(JSON.parse(request.responseText) as T);
    };
    request.onerror = () => reject(new Error("Audio upload failed"));
    request.send(body);
  });
}

function uploadProgress(
  file: File,
  fileIndex: number,
  fileCount: number,
  filePercent: number,
  phase: AudioUploadProgress["phase"],
): AudioUploadProgress {
  const safeFilePercent = Math.max(0, Math.min(100, filePercent));
  return {
    fileName: file.name,
    fileIndex,
    fileCount,
    filePercent: safeFilePercent,
    overallPercent: ((fileIndex + safeFilePercent / 100) / fileCount) * 100,
    phase,
  };
}

function waveformPayload(buffer: AudioBuffer): string {
  return JSON.stringify({
    sample_rate: buffer.sampleRate,
    points_per_second: 100,
    peaks: waveformPeaks(buffer, 100),
  });
}

function waveformPeaks(buffer: AudioBuffer, pointsPerSecond: number): [number, number][] {
  const step = Math.max(1, Math.floor(buffer.sampleRate / pointsPerSecond));
  const points = Math.ceil(buffer.length / step);
  const peaks: [number, number][] = [];
  for (let point = 0; point < points; point += 1) {
    let minimum = 1;
    let maximum = -1;
    const start = point * step;
    const end = Math.min(buffer.length, start + step);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const data = buffer.getChannelData(channel);
      for (let index = start; index < end; index += 1) {
        const value = data[index] ?? 0;
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
    }
    peaks.push([roundPeak(minimum), roundPeak(maximum)]);
  }
  return peaks;
}

function roundPeak(value: number): number {
  return Math.round(value * 1000) / 1000;
}
