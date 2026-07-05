import { VOICE_COUNT, getVoiceRow } from "@/mock/data";
import type { Voice } from "@/mock/types";

export type VoiceSort = "name" | "segments" | "segments_asc";
export type VoiceQuery = {
  query: string;
  dataset: string;
  minSegments: number;
  sort: VoiceSort;
};

/**
 * Mock server-owned voice table. Filtering/sorting happens HERE (server-side),
 * not in the browser — the frontend must never scan the full set, which can be
 * in the millions. A real backend owns and indexes this; the client only ever
 * receives an already-filtered page.
 *
 * ponytail: bounded to VOICE_COUNT for the mock; the real service pages millions.
 */
const SERVER: Voice[] = Array.from({ length: VOICE_COUNT }, (_, i) => getVoiceRow(i));

function applyQuery(list: Voice[], p: VoiceQuery): Voice[] {
  const q = p.query.trim().toLowerCase();
  const out = list.filter(
    (v) =>
      (!q || v.name.toLowerCase().includes(q)) &&
      v.segments >= p.minSegments &&
      (p.dataset === "all" || v.datasets.includes(p.dataset)),
  );
  out.sort((a, b) =>
    p.sort === "segments"
      ? b.segments - a.segments
      : p.sort === "segments_asc"
        ? a.segments - b.segments
        : a.name.localeCompare(b.name),
  );
  return out;
}

const delay = <T>(value: T): Promise<T> =>
  new Promise((resolve) => setTimeout(() => resolve(value), 250));

export function fetchVoices(params: VoiceQuery): Promise<{ rows: Voice[]; total: number }> {
  return delay({ rows: applyQuery(SERVER, params), total: SERVER.length });
}

export function createVoice(): Promise<string> {
  const name = `speaker_${SERVER.length + 1}`;
  SERVER.unshift({ id: `v_${Date.now()}`, name, segments: 0, datasets: [] });
  return delay(name);
}

export function renameVoice(id: string, name: string): Promise<void> {
  const v = SERVER.find((x) => x.id === id);
  if (v) v.name = name;
  return delay(undefined);
}

export function deleteVoice(id: string): Promise<void> {
  const i = SERVER.findIndex((x) => x.id === id);
  if (i >= 0) SERVER.splice(i, 1);
  return delay(undefined);
}
