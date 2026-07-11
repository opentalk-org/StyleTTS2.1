import { create } from "zustand";

type JobsSelectionStore = {
  selection: Record<string, true>;
  limit: number;
  offset: number;
  setLimit: (limit: number) => void;
  setOffset: (offset: number) => void;
  toggleSelect: (id: string) => void;
  selectMany: (ids: string[]) => void;
  clearSelection: () => void;
};

export const useJobsSelection = create<JobsSelectionStore>((set) => ({
  selection: {},
  limit: 100,
  offset: 0,
  setLimit: (limit) => set({ limit }),
  setOffset: (offset) => set({ offset }),
  toggleSelect: (id) =>
    set((s) => {
      const selection = { ...s.selection };
      if (selection[id]) delete selection[id];
      else selection[id] = true;
      return { selection };
    }),
  selectMany: (ids) =>
    set((s) => {
      const selection = { ...s.selection };
      for (const id of ids) selection[id] = true;
      return { selection };
    }),
  clearSelection: () => set({ selection: {} }),
}));
