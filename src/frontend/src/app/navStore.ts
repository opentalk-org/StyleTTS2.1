import { create } from "zustand";
import { persist } from "zustand/middleware";

import { defaultBackendUrl } from "./backendConfig";

export type Screen =
  | "datasets"
  | "voices"
  | "audio"
  | "editor"
  | "mos"
  | "statistics"
  | "artifacts"
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
  activeAudioFileId: string | null;
  go: (screen: Screen) => void;
  toggleNav: () => void;
  setBackendUrl: (url: string) => void;
  connect: () => void;
  openEditor: (audioFileId: string) => void;
};

export const useNav = create<NavStore>()(
  persist(
    (set) => ({
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
    }),
    {
      // Restore the last-open page and backend on reload instead of dropping back
      // to the Connect screen.
      name: "styletts-nav",
      partialize: (s) => ({
        screen: s.screen,
        navCollapsed: s.navCollapsed,
        connected: s.connected,
        backendUrl: s.backendUrl,
        activeAudioFileId: s.activeAudioFileId,
      }),
    },
  ),
);
