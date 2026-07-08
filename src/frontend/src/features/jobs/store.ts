import { create } from "zustand";

type JobsSelectionStore = {
  selection: Record<string, true>;
  toggleSelect: (id: string) => void;
  selectMany: (ids: string[]) => void;
  clearSelection: () => void;
};

export const useJobsSelection = create<JobsSelectionStore>((set) => ({
  selection: {},
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
