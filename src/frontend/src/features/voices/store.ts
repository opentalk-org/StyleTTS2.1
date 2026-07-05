import { create } from "zustand";

import { VOICE_COUNT, getVoiceRow } from "../../mock/data";
import type { Voice } from "../../mock/types";

export type PreviewState = "ready" | "loading" | "error";

type VoicesStore = {
  voices: Voice[];
  preview: PreviewState;
  query: string;
  dataset: string;
  minSegments: number;
  sort: "name" | "segments" | "segments_asc";
  editId: string | null;
  add: () => void;
  rename: (id: string, name: string) => void;
  remove: (id: string) => void;
  set: (patch: Partial<VoicesStore>) => void;
};

const seed = Array.from({ length: VOICE_COUNT }, (_, i) => getVoiceRow(i));

export const useVoices = create<VoicesStore>((set) => ({
  voices: seed,
  preview: "ready",
  query: "",
  dataset: "all",
  minSegments: 0,
  sort: "name",
  editId: null,
  add: () =>
    set((s) => {
      const name = `speaker_${s.voices.length + 1}`;
      return { voices: [{ id: `v_${Date.now()}`, name, segments: 0, datasets: [] }, ...s.voices] };
    }),
  rename: (id, name) =>
    set((s) => ({ voices: s.voices.map((v) => (v.id === id ? { ...v, name } : v)) })),
  remove: (id) => set((s) => ({ voices: s.voices.filter((v) => v.id !== id) })),
  set: (patch) => set(patch),
}));

/** Apply the current filters/sort to the voice list. */
export function filterVoices(s: VoicesStore): Voice[] {
  const q = s.query.trim().toLowerCase();
  const list = s.voices.filter(
    (v) =>
      (!q || v.name.toLowerCase().includes(q)) &&
      v.segments >= s.minSegments &&
      (s.dataset === "all" || v.datasets.includes(s.dataset)),
  );
  list.sort((a, b) =>
    s.sort === "segments"
      ? b.segments - a.segments
      : s.sort === "segments_asc"
        ? a.segments - b.segments
        : a.name.localeCompare(b.name),
  );
  return list;
}
