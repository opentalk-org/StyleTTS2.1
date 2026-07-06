import { create } from "zustand";

export type AudioSort = "updated" | "name" | "duration" | "speaker" | "segments";

export type AudioFilters = {
  query: string;
  dataset: string;
  sort: AudioSort;
};

type AudioStore = AudioFilters & {
  limit: number;
  offset: number;
  /** file id -> selected. */
  selection: Record<string, true>;
  /** "select all N matching the current filter" mode (server-side selection). */
  selectAllMatching: boolean;
  /** file id -> expanded inline-segments preview. */
  expanded: Record<string, true>;
  setFilters: (patch: Partial<AudioFilters> & Partial<Pick<AudioStore, "limit" | "offset">>) => void;
  toggleSelect: (id: string) => void;
  clearSelection: () => void;
  selectAllFiltered: () => void;
  toggleExpanded: (id: string) => void;
};

export const useAudio = create<AudioStore>((set) => ({
  query: "",
  dataset: "all",
  sort: "updated",
  limit: 100,
  offset: 0,
  selection: {},
  selectAllMatching: false,
  expanded: {},
  // Changing a filter resets the "all matching" mode — it no longer describes the visible set.
  setFilters: (patch) => set({ ...patch, selectAllMatching: false }),
  toggleSelect: (id) =>
    set((s) => {
      const selection = { ...s.selection };
      if (selection[id]) delete selection[id];
      else selection[id] = true;
      return { selection, selectAllMatching: false };
    }),
  clearSelection: () => set({ selection: {}, selectAllMatching: false }),
  selectAllFiltered: () => set({ selectAllMatching: true }),
  toggleExpanded: (id) =>
    set((s) => {
      const expanded = { ...s.expanded };
      if (expanded[id]) delete expanded[id];
      else expanded[id] = true;
      return { expanded };
    }),
}));
