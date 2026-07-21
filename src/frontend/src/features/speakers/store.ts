import { create } from "zustand";

export type SpeakerFilters = {
  query: string;
  limit: number;
  offset: number;
  editId: string | null;
};

type SpeakerFiltersStore = SpeakerFilters & {
  selection: Record<string, true>;
  selectAllMatching: boolean;
  visibleIds: string[];
  set: (patch: Partial<SpeakerFilters>) => void;
  setVisibleIds: (ids: string[]) => void;
  toggleSelect: (id: string) => void;
  selectVisible: () => void;
  clearSelection: () => void;
  selectAllFiltered: () => void;
};

/**
 * View state for the Speakers screen (filter inputs + inline-rename target +
 * multi-selection). The filter values are sent to the server query — they are
 * NOT used to filter a client-side array. See api.ts.
 *
 * Selection mirrors the audio tab: an explicit id map, plus a `selectAllMatching`
 * flag for "select every speaker matching the current filter, across pages".
 * Changing the search query clears `selectAllMatching` since the matching set moved.
 */
export const useSpeakerFilters = create<SpeakerFiltersStore>((set) => ({
  query: "",
  limit: 100,
  offset: 0,
  editId: null,
  selection: {},
  selectAllMatching: false,
  visibleIds: [],
  set: (patch) =>
    set((s) => ({ ...patch, selectAllMatching: "query" in patch ? false : s.selectAllMatching })),
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
}));
