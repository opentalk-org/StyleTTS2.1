import { create } from "zustand";

import type { CorpusTab } from "./logic";

type StatisticsUiStore = {
  entryId: string | null;
  tab: CorpusTab;
  setEntryId: (id: string | null) => void;
  setTab: (tab: CorpusTab) => void;
};

export const useStatisticsUi = create<StatisticsUiStore>((set) => ({
  entryId: null,
  tab: "transcript",
  setEntryId: (entryId) => set({ entryId }),
  setTab: (tab) => set({ tab }),
}));
