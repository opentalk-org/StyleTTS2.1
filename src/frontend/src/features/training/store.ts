import { create } from "zustand";

import type { WorkflowGraph } from "../workflows/types";

export type TrainTab = "styletts" | "f0" | "asr" | "mos";

type TrainingStore = {
  trainTab: TrainTab;
  graphsByTab: Partial<Record<TrainTab, WorkflowGraph>>;
  setTrainTab: (tab: TrainTab) => void;
  ensureGraph: (tab: TrainTab, graph: WorkflowGraph) => void;
  setGraph: (tab: TrainTab, graph: WorkflowGraph) => void;
};

export const useTraining = create<TrainingStore>((set) => ({
  trainTab: "styletts",
  graphsByTab: {},
  setTrainTab: (trainTab) => set({ trainTab }),
  ensureGraph: (tab, graph) =>
    set((s) => (s.graphsByTab[tab] ? {} : { graphsByTab: { ...s.graphsByTab, [tab]: graph } })),
  setGraph: (tab, graph) => set((s) => ({ graphsByTab: { ...s.graphsByTab, [tab]: graph } })),
}));
