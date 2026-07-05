import { create } from "zustand";

import { showToast } from "@/shared/feedback/Toast";
import { phonemize, synthDuration } from "./logic";

export type TestMode = "single" | "sweep";

export type SingleConfig = {
  ckpt: string;
  weights: string;
  text: string;
  lang: string;
  steps: number;
  emb: number;
  styleRef: string;
  styleMix: number;
  prosodyMix: number;
};

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

export type SweepConfig = {
  text: string;
  voices: Record<string, boolean>;
  n: number;
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
  single: SingleConfig;
  testResults: TestResult[];
  sweep: SweepConfig;
  sweepResults: SweepResult[];
  setMode: (mode: TestMode) => void;
  setSingle: <K extends keyof SingleConfig>(key: K, value: SingleConfig[K]) => void;
  setSweep: <K extends keyof SweepConfig>(key: K, value: SweepConfig[K]) => void;
  toggleVoice: (voice: string) => void;
  genSingle: () => void;
  genSweep: () => void;
};

function id(prefix: string): string {
  return prefix + Math.random().toString(16).slice(2, 6);
}

export const useTesting = create<TestingStore>((set, get) => ({
  testMode: "single",
  single: {
    ckpt: "",
    weights: "",
    text: "The quiet hum of the studio settled as the first take began.",
    lang: "en-us",
    steps: 5,
    emb: 1.0,
    styleRef: "sr_1",
    styleMix: 0.7,
    prosodyMix: 0.5,
  },
  testResults: [],
  sweep: {
    text: "Welcome back to the show — today we have a lot to cover.",
    voices: {},
    n: 2,
  },
  sweepResults: [],

  setMode: (mode) => set({ testMode: mode }),
  setSingle: (key, value) => set((s) => ({ single: { ...s.single, [key]: value } })),
  setSweep: (key, value) => set((s) => ({ sweep: { ...s.sweep, [key]: value } })),
  toggleVoice: (voice) =>
    set((s) => ({ sweep: { ...s.sweep, voices: { ...s.sweep.voices, [voice]: !s.sweep.voices[voice] } } })),

  genSingle: () => {
    const c = get().single;
    if (!c.ckpt) {
      showToast("Select a checkpoint first", undefined, "error");
      return;
    }
    const rid = id("syn_");
    const result: TestResult = {
      id: rid,
      text: c.text,
      phon: phonemize(c.text),
      dur: synthDuration(rid),
      steps: c.steps,
      emb: c.emb,
      when: Date.now(),
      file: rid + ".wav",
    };
    set((s) => ({ testResults: [result, ...s.testResults] }));
    showToast("Synthesis complete", rid);
  },

  genSweep: () => {
    const sw = get().sweep;
    const voices = Object.keys(sw.voices).filter((v) => sw.voices[v]);
    if (!voices.length) {
      showToast("Select at least one voice", undefined, "error");
      return;
    }
    const out: SweepResult[] = [];
    for (const voice of voices) {
      for (let i = 0; i < sw.n; i++) {
        const rid = id("sw_");
        out.push({ id: rid, voice, sample: i + 1, dur: synthDuration(rid, i), file: rid + ".wav" });
      }
    }
    set({ sweepResults: out });
    showToast("Sweep complete", `${out.length} samples`);
  },
}));
