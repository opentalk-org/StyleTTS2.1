import type { Tone } from "@/shared/ui/Badge";
import type { Checkpoint, CheckpointType } from "@/mock/types";
import type { CheckpointsStore } from "./store";

export const TYPE_TONE: Record<CheckpointType, Tone> = {
  styletts2: "blue",
  asr: "emerald",
  f0: "amber",
  plbert: "gray",
};

/** Filter by query + type, then group by originating training job. */
export function groupCheckpoints(s: CheckpointsStore): Record<string, Checkpoint[]> {
  const q = s.query.trim().toLowerCase();
  const list = s.checkpoints.filter(
    (c) => (!q || c.name.toLowerCase().includes(q)) && (s.type === "all" || c.type === s.type),
  );
  const groups: Record<string, Checkpoint[]> = {};
  for (const c of list) (groups[c.job] ??= []).push(c);
  return groups;
}

export const CATALOG: { name: string; file: string; size: string }[] = [
  { name: "StyleTTS2 · LibriTTS", file: "styletts2_libritts.pth", size: "1.2 GB" },
  { name: "StyleTTS2 · LJSpeech", file: "styletts2_ljspeech.pth", size: "842 MB" },
  { name: "PL-BERT · multilingual", file: "plbert_ml.t7", size: "420 MB" },
  { name: "ASR · base aligner", file: "asr_base.pth", size: "310 MB" },
];
