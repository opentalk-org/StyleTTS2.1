import { create } from "zustand";

export type VoiceFilters = {
  query: string;
  limit: number;
  offset: number;
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
  limit: 100,
  offset: 0,
  editId: null,
  set: (patch) => set(patch),
}));
