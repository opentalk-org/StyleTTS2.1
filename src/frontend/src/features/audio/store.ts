import { create } from "zustand";

export type AudioSort = "updated" | "name" | "duration" | "speaker_id" | "segments";

export type AudioFilters = {
  query: string;
  language: string;
  dataset: string;
  sort: AudioSort;
};

type AudioStore = AudioFilters & {
  limit: number;
  offset: number;
  selection: Record<string, true>;
  selectAllMatching: boolean;
  visibleIds: string[];
  expanded: Record<string, true>;
  setFilters: (patch: Partial<AudioFilters> & Partial<Pick<AudioStore, "limit" | "offset">>) => void;
  setVisibleIds: (ids: string[]) => void;
  toggleSelect: (id: string) => void;
  selectVisible: () => void;
  clearSelection: () => void;
  selectAllFiltered: () => void;
  toggleExpanded: (id: string) => void;
};

export const useAudio = create<AudioStore>((set) => ({
  query: "",
  language: "",
  dataset: "all",
  sort: "updated",
  limit: 100,
  offset: 0,
  selection: {},
  selectAllMatching: false,
  visibleIds: [],
  expanded: {},
  setFilters: (patch) => set({ ...patch, selectAllMatching: false }),
  setVisibleIds: (ids) => set({ visibleIds: ids }),
  toggleSelect: (id) =>
    set((s) => {
      if (s.selectAllMatching) {
        const selection: Record<string, true> = {};
        for (const visibleId of s.visibleIds) if (visibleId !== id) selection[visibleId] = true;
        return { selection, selectAllMatching: false };
      }
      const selection = { ...s.selection };
      if (selection[id]) delete selection[id];
      else selection[id] = true;
      return { selection };
    }),
  selectVisible: () =>
    set((s) => {
      const selection: Record<string, true> = {};
      for (const id of s.visibleIds) selection[id] = true;
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
