import { create } from "zustand";

import { seedCheckpoints } from "@/mock/data";
import type { Checkpoint } from "@/mock/types";

export type CheckpointsStore = {
  checkpoints: Checkpoint[];
  query: string;
  type: string;
  rename: (id: string, name: string) => void;
  remove: (id: string) => void;
  set: (patch: Partial<CheckpointsStore>) => void;
};

export const useCheckpoints = create<CheckpointsStore>((set) => ({
  checkpoints: seedCheckpoints(),
  query: "",
  type: "all",
  rename: (id, name) =>
    set((s) => ({ checkpoints: s.checkpoints.map((c) => (c.id === id ? { ...c, name } : c)) })),
  remove: (id) => set((s) => ({ checkpoints: s.checkpoints.filter((c) => c.id !== id) })),
  set: (patch) => set(patch),
}));
