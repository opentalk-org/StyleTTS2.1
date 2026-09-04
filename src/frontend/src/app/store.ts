import { create } from "zustand";
import { persist } from "zustand/middleware";

import { defaultBackendUrl } from "./backendConfig";

type AppStore = {
  navCollapsed: boolean;
  connected: boolean;
  backendUrl: string;
  toggleNav: () => void;
  setBackendUrl: (url: string) => void;
  connect: () => void;
};

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      navCollapsed: false,
      connected: false,
      backendUrl: defaultBackendUrl(),
      toggleNav: () => set((state) => ({ navCollapsed: !state.navCollapsed })),
      setBackendUrl: (backendUrl) => set({ backendUrl }),
      connect: () => set({ connected: true }),
    }),
    {
      name: "styletts-app",
      version: 1,
      partialize: (state) => ({
        navCollapsed: state.navCollapsed,
        connected: state.connected,
        backendUrl: state.backendUrl,
      }),
    },
  ),
);
