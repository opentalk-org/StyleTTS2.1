import { create } from "zustand";

type CheckpointsFilterStore = {
  query: string;
  type: string;
  setQuery: (query: string) => void;
  setType: (type: string) => void;
};

export const useCheckpointsFilters = create<CheckpointsFilterStore>((set) => ({
  query: "",
  type: "all",
  setQuery: (query) => set({ query }),
  setType: (type) => set({ type }),
}));
