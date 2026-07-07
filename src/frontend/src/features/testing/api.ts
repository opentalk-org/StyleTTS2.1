import { backendRequest } from "@/app/backend";

import type { AudioFile } from "../audio/api";

export function fetchRunAudioFiles(runId: string): Promise<AudioFile[]> {
  return backendRequest<AudioFile[]>(`/audio-files/by-run/${encodeURIComponent(runId)}`);
}
