import { create } from "zustand";
import { persist } from "zustand/middleware";

type FrontendPreferences = {
  confirmDeletes: boolean;
  setConfirmDeletes: (confirmDeletes: boolean) => void;
};

export const useFrontendPreferences = create<FrontendPreferences>()(
  persist(
    (set) => ({
      confirmDeletes: true,
      setConfirmDeletes: (confirmDeletes) => set({ confirmDeletes }),
    }),
    { name: "runflow-frontend-preferences" },
  ),
);
