import { create } from "zustand";

type MosStore = {
  selectedDatasetIds: string[];
  scoreA: string;
  scoreB: string;
  preferredAudioId: string | null;
  toggleDataset: (datasetId: string) => void;
  setScoreA: (score: string) => void;
  setScoreB: (score: string) => void;
  setPreferredAudioId: (audioId: string) => void;
  resetPair: (scoreA: string, scoreB: string) => void;
};

export const useMos = create<MosStore>((set) => ({
  selectedDatasetIds: [],
  scoreA: "",
  scoreB: "",
  preferredAudioId: null,
  toggleDataset: (datasetId) =>
    set((state) => ({
      selectedDatasetIds: state.selectedDatasetIds.includes(datasetId)
        ? state.selectedDatasetIds.filter((id) => id !== datasetId)
        : [...state.selectedDatasetIds, datasetId],
    })),
  setScoreA: (scoreA) => set({ scoreA }),
  setScoreB: (scoreB) => set({ scoreB }),
  setPreferredAudioId: (preferredAudioId) => set({ preferredAudioId }),
  resetPair: (scoreA, scoreB) => set({ scoreA, scoreB, preferredAudioId: null }),
}));
