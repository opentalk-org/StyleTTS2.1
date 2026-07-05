import { create } from "zustand";

type Settings = {
  autoNormalize: boolean;
  confirmDeletes: boolean;
  pollWhenIdle: boolean;
  theme: "light" | "system" | "dark";
  defaultLang: string;
};

type SettingsStore = Settings & {
  set: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
};

export const useSettings = create<SettingsStore>((set) => ({
  autoNormalize: true,
  confirmDeletes: true,
  pollWhenIdle: false,
  theme: "light",
  defaultLang: "en-us",
  set: (key, value) => set({ [key]: value } as Partial<SettingsStore>),
}));
