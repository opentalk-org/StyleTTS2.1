import { create } from "zustand";

import { seedDatasets } from "@/mock/data";
import type { Dataset } from "@/mock/types";

type DatasetsStore = {
  datasets: Dataset[];
  create: (name: string) => void;
  remove: (id: string) => void;
};

/** Mock datasets backend — in a real app these would be server-owned. */
export const useDatasets = create<DatasetsStore>((set) => ({
  datasets: seedDatasets(),
  create: (name) =>
    set((s) => ({
      datasets: [...s.datasets, { id: `ds_${Date.now()}`, name, files: 0 }],
    })),
  remove: (id) => set((s) => ({ datasets: s.datasets.filter((d) => d.id !== id) })),
}));
