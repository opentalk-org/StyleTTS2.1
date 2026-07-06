import { create } from "zustand";

import { defaultBackendUrl } from "./backendConfig";

export type Screen =
  | "datasets"
  | "voices"
  | "audio"
  | "editor"
  | "statistics"
  | "workflows"
  | "checkpoints"
  | "training"
  | "runs"
  | "testing"
  | "cluster"
  | "jobs"
  | "settings";

type NavStore = {
  screen: Screen;
  navCollapsed: boolean;
  connected: boolean;
  backendUrl: string;
  /** Set when opening the segment editor from the Audio Files table. */
  activeAudioFileId: string | null;
  go: (screen: Screen) => void;
  toggleNav: () => void;
  setBackendUrl: (url: string) => void;
  connect: () => void;
  openEditor: (audioFileId: string) => void;
};

export const useNav = create<NavStore>((set) => ({
  screen: "training",
  navCollapsed: false,
  connected: false,
  backendUrl: defaultBackendUrl(),
  activeAudioFileId: null,
  go: (screen) => set({ screen }),
  toggleNav: () => set((s) => ({ navCollapsed: !s.navCollapsed })),
  setBackendUrl: (backendUrl) => set({ backendUrl }),
  connect: () => set({ connected: true }),
  openEditor: (activeAudioFileId) => set({ screen: "editor", activeAudioFileId }),
}));
