import { backendRequest } from "@/app/backend";

export type PiperVoiceCatalogItem = {
  key: string;
  name: string;
  language: { code: string; name_english: string; country_english: string };
  quality: string;
};

export function fetchPiperVoices(path: string, signal: AbortSignal): Promise<Record<string, PiperVoiceCatalogItem>> {
  return backendRequest(path, { signal });
}
