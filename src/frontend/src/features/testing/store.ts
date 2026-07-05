import { create } from "zustand";

import { showToast } from "@/shared/feedback/Toast";
import { phonemize, synthDuration, type SingleConfig, type SweepConfig, type TestingMode } from "./logic";

export type TestMode = TestingMode;

export type TestResult = {
  id: string;
  text: string;
  phon: string;
  dur: number;
  steps: number;
  emb: number;
  when: number;
  file: string;
};

export type SweepResult = {
  id: string;
  voice: string;
  sample: number;
  dur: number;
  file: string;
};

export type TestingStore = {
  testMode: TestMode;
  testResults: TestResult[];
  sweepResults: SweepResult[];
  setMode: (mode: TestMode) => void;
  genSingle: (config: SingleConfig) => void;
  genSweep: (config: SweepConfig) => void;
};

function id(prefix: string): string {
  return prefix + Math.random().toString(16).slice(2, 6);
}

export const useTesting = create<TestingStore>((set) => ({
  testMode: "single",
  testResults: [],
  sweepResults: [],

  setMode: (mode) => set({ testMode: mode }),

  genSingle: (c) => {
    if (!c.ckpt) {
      showToast("Select a checkpoint first", undefined, "error");
      return;
    }
    if (!c.alphabetSymbols.trim()) {
      showToast("Set a phoneme alphabet first", undefined, "error");
      return;
    }
    const rid = id("syn_");
    const result: TestResult = {
      id: rid,
      text: c.text,
      phon: phonemize(c.text, c.alphabetSymbols),
      dur: synthDuration(rid),
      steps: c.steps,
      emb: c.emb,
      when: Date.now(),
      file: rid + ".wav",
    };
    set((s) => ({ testResults: [result, ...s.testResults] }));
    showToast("Synthesis complete", rid);
  },

  genSweep: (sw) => {
    if (!sw.ckpt) {
      showToast("Select a checkpoint first", undefined, "error");
      return;
    }
    if (!sw.voices.length) {
      showToast("Select at least one voice", undefined, "error");
      return;
    }
    if (!sw.alphabetSymbols.trim()) {
      showToast("Set a phoneme alphabet first", undefined, "error");
      return;
    }
    const out: SweepResult[] = [];
    for (const voice of sw.voices) {
      for (let i = 0; i < sw.n; i++) {
        const rid = id("sw_");
        out.push({ id: rid, voice, sample: i + 1, dur: synthDuration(rid, i), file: rid + ".wav" });
      }
    }
    set({ sweepResults: out });
    showToast("Sweep complete", `${out.length} samples`);
  },
}));
