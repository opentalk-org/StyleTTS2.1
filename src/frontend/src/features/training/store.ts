import { create } from "zustand";

export type TrainTab = "styletts" | "f0" | "asr" | "mos";

type TrainingStore = {
  trainTab: TrainTab;
  setTrainTab: (tab: TrainTab) => void;
};

export const useTraining = create<TrainingStore>((set) => ({
  trainTab: "styletts",
  setTrainTab: (trainTab) => set({ trainTab }),
}));
