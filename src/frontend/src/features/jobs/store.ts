import { create } from "zustand";

import { seedJobs } from "../../mock/jobs";
import type { Job, JobType } from "../../mock/types";

type JobsStore = {
  jobs: Job[];
  startJob: (type: JobType, label: string) => string;
  rerun: (id: string) => void;
  kill: (id: string) => void;
  remove: (id: string) => void;
  /** Advance running jobs; called on an interval by the app shell. */
  tick: () => void;
};

export const useJobs = create<JobsStore>((set) => ({
  jobs: seedJobs(),
  startJob: (type, label) => {
    const id = `job_${Math.random().toString(16).slice(2, 6)}`;
    set((s) => ({
      jobs: [
        { id, type, label, status: "queued", progress: 0, updated: Date.now() },
        ...s.jobs,
      ],
    }));
    return id;
  },
  rerun: (id) =>
    set((s) => ({
      jobs: s.jobs.map((j) =>
        j.id === id ? { ...j, status: "queued", progress: 0, error: undefined, stale: false, updated: Date.now() } : j,
      ),
    })),
  kill: (id) =>
    set((s) => ({
      jobs: s.jobs.map((j) =>
        j.id === id ? { ...j, status: "failed", error: "Killed by user.", stale: false, updated: Date.now() } : j,
      ),
    })),
  remove: (id) => set((s) => ({ jobs: s.jobs.filter((j) => j.id !== id) })),
  tick: () =>
    set((s) => ({
      jobs: s.jobs.map((j) => {
        if (j.status === "queued") return { ...j, status: "running", updated: Date.now() };
        if (j.status !== "running" || j.stale) return j;
        const progress = j.progress + (2 + Math.random() * 5);
        if (progress >= 100)
          return { ...j, status: "succeeded", progress: 100, updated: Date.now() };
        return {
          ...j,
          progress,
          updated: Date.now(),
          ...(j.type === "training" && j.epoch
            ? { step: (j.step ?? 0) + 40 }
            : {}),
        };
      }),
    })),
}));

export function runningCount(jobs: Job[]): number {
  return jobs.filter((j) => j.status === "running").length;
}
