import { create } from "zustand";

import { showToast } from "@/shared/feedback/Toast";
import { fetchRun, startGraph } from "../workflows/api";
import type { WorkflowPayload } from "../workflows/types";
import { fetchRunAudioFiles } from "./api";
import type { TestingMode } from "./logic";

export type TestMode = TestingMode;

export type RunState = "running" | "succeeded" | "failed";

export type TestResult = {
  id: string;
  runId: string;
  state: RunState;
  text: string;
  steps: number;
  emb: number;
  when: number;
  audioFileId?: string;
  duration?: number;
  name?: string;
  error?: string;
};

export type SweepResult = {
  id: string;
  runId: string;
  state: RunState;
  voice: string;
  sample: number;
  audioFileId?: string;
  duration?: number;
  name?: string;
  error?: string;
};

export type SingleDisplay = { text: string; steps: number; emb: number };
export type SweepDisplay = { voice: string; sample: number };

export type TestingStore = {
  testMode: TestMode;
  testResults: TestResult[];
  sweepResults: SweepResult[];
  setMode: (mode: TestMode) => void;
  runSingle: (payload: WorkflowPayload, display: SingleDisplay) => Promise<void>;
  runSweep: (payload: WorkflowPayload, display: SweepDisplay[]) => Promise<void>;
};

const TERMINAL = new Set(["succeeded", "failed", "stopped"]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollRun(runId: string): Promise<string> {
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    const run = await fetchRun(runId);
    if (TERMINAL.has(run.state)) return run.error ? `failed:${run.error}` : run.state;
    await sleep(1500);
  }
  return "failed:timed out";
}

export const useTesting = create<TestingStore>((set) => ({
  testMode: "single",
  testResults: [],
  sweepResults: [],

  setMode: (mode) => set({ testMode: mode }),

  runSingle: async (payload, display) => {
    let runId: string;
    try {
      const run = await startGraph(payload);
      runId = run.run_id;
    } catch (error) {
      showToast("Could not start synthesis", String(error), "error");
      return;
    }
    const result: TestResult = { id: runId, runId, state: "running", when: Date.now(), ...display };
    set((s) => ({ testResults: [result, ...s.testResults] }));
    const patch = (next: Partial<TestResult>) =>
      set((s) => ({ testResults: s.testResults.map((r) => (r.id === runId ? { ...r, ...next } : r)) }));
    try {
      const outcome = await pollRun(runId);
      if (outcome !== "succeeded") {
        patch({ state: "failed", error: outcome.replace(/^failed:/, "") });
        showToast("Synthesis failed", outcome.replace(/^failed:/, ""), "error");
        return;
      }
      const audios = await fetchRunAudioFiles(runId);
      const audio = audios[audios.length - 1];
      if (!audio) {
        patch({ state: "failed", error: "No audio produced" });
        showToast("No audio produced", undefined, "error");
        return;
      }
      patch({ state: "succeeded", audioFileId: audio.id, duration: audio.duration, name: audio.name });
      showToast("Synthesis complete", runId);
    } catch (error) {
      patch({ state: "failed", error: String(error) });
      showToast("Synthesis failed", String(error), "error");
    }
  },

  runSweep: async (payload, display) => {
    let runId: string;
    try {
      const run = await startGraph(payload);
      runId = run.run_id;
    } catch (error) {
      showToast("Could not start sweep", String(error), "error");
      return;
    }
    const pending: SweepResult[] = display.map((item, index) => ({
      id: `${runId}:${index}`,
      runId,
      state: "running",
      voice: item.voice,
      sample: item.sample,
    }));
    set({ sweepResults: pending });
    const finalize = (next: (item: SweepResult, index: number) => SweepResult) =>
      set((s) => ({ sweepResults: s.sweepResults.map((item, index) => (item.runId === runId ? next(item, index) : item)) }));
    try {
      const outcome = await pollRun(runId);
      if (outcome !== "succeeded") {
        finalize((item) => ({ ...item, state: "failed", error: outcome.replace(/^failed:/, "") }));
        showToast("Sweep failed", outcome.replace(/^failed:/, ""), "error");
        return;
      }
      const audios = await fetchRunAudioFiles(runId);
      finalize((item, index) => {
        const audio = audios[index];
        if (!audio) return { ...item, state: "failed", error: "No audio produced" };
        return { ...item, state: "succeeded", audioFileId: audio.id, duration: audio.duration, name: audio.name };
      });
      showToast("Sweep complete", `${audios.length} samples`);
    } catch (error) {
      finalize((item) => ({ ...item, state: "failed", error: String(error) }));
      showToast("Sweep failed", String(error), "error");
    }
  },
}));
