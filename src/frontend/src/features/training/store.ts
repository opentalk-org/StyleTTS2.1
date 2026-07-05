import { create } from "zustand";

export type TrainTab = "styletts" | "f0" | "asr";

export type OodSet = { id: string; name: string; n: number };

export type Toggles = {
  multispeaker: boolean;
  stagewise: boolean;
  mixedprec: boolean;
};

type Draft = {
  trainTab: TrainTab;
  alphabet: string;
  seqSeconds: number;
  toggles: Toggles;
  oodSets: OodSet[];
};

type TrainingStore = Draft & {
  setTrainTab: (tab: TrainTab) => void;
  setAlphabet: (alphabet: string) => void;
  setSeqSeconds: (seqSeconds: number) => void;
  setToggle: (key: keyof Toggles, value: boolean) => void;
  addOod: (name: string, n: number) => void;
  removeOod: (id: string) => void;
};

const STORAGE_KEY = "stts_draft";

const DEFAULT_DRAFT: Draft = {
  trainTab: "styletts",
  alphabet:
    "a b c d e f g h i j k l m n o p q r s t u v w x y z ɑ æ ə ɛ ɪ ʊ ʌ ɔ θ ð ʃ ʒ ŋ tʃ dʒ aɪ aʊ eɪ oʊ ɔɪ ɝ ɚ ˈ ˌ ː . , ? ! ' \" ( ) -",
  seqSeconds: 8.0,
  toggles: { multispeaker: true, stagewise: true, mixedprec: false },
  oodSets: [
    { id: "ood_1", name: "librispeech_eval.txt", n: 512 },
    { id: "ood_2", name: "in_domain_prompts.txt", n: 128 },
  ],
};

function hydrate(): Draft {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return DEFAULT_DRAFT;
  return { ...DEFAULT_DRAFT, ...(JSON.parse(raw) as Partial<Draft>) };
}

function persist(state: Draft) {
  const draft: Draft = {
    trainTab: state.trainTab,
    alphabet: state.alphabet,
    seqSeconds: state.seqSeconds,
    toggles: state.toggles,
    oodSets: state.oodSets,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
}

export const useTraining = create<TrainingStore>((set) => ({
  ...hydrate(),
  setTrainTab: (trainTab) => set({ trainTab }),
  setAlphabet: (alphabet) => set({ alphabet }),
  setSeqSeconds: (seqSeconds) => set({ seqSeconds }),
  setToggle: (key, value) =>
    set((s) => ({ toggles: { ...s.toggles, [key]: value } })),
  addOod: (name, n) =>
    set((s) => ({
      oodSets: [...s.oodSets, { id: `ood_${Date.now()}`, name, n }],
    })),
  removeOod: (id) =>
    set((s) => ({ oodSets: s.oodSets.filter((x) => x.id !== id) })),
}));

useTraining.subscribe(persist);
