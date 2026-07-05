import { create } from "zustand";

import type { VoiceSort } from "./api";

export type VoiceFilters = {
  query: string;
  dataset: string;
  minSegments: number;
  sort: VoiceSort;
  editId: string | null;
};

type VoiceFiltersStore = VoiceFilters & {
  set: (patch: Partial<VoiceFilters>) => void;
};

/**
 * View state for the Voices screen (filter inputs + inline-rename target).
 * The filter values are sent to the server query — they are NOT used to filter
 * a client-side array. See api.ts.
 */
export const useVoiceFilters = create<VoiceFiltersStore>((set) => ({
  query: "",
  dataset: "all",
  minSegments: 0,
  sort: "name",
  editId: null,
  set: (patch) => set(patch),
}));
