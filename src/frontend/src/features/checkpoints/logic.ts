import type { Tone } from "@/shared/ui/Badge";
import type { Checkpoint } from "./api";

export const TYPE_TONE: Record<string, Tone> = {
  styletts2: "blue",
  asr: "emerald",
  f0: "amber",
  plbert: "gray",
};

export function checkpointTone(type: string): Tone {
  return TYPE_TONE[type] ?? "gray";
}

export function groupCheckpoints(items: Checkpoint[], query: string, type: string): Record<string, Checkpoint[]> {
  const q = query.trim().toLowerCase();
  const list = items.filter(
    (c) => (!q || c.name.toLowerCase().includes(q)) && (type === "all" || c.type_ === type),
  );
  const groups: Record<string, Checkpoint[]> = {};
  for (const c of list) (groups[c.job_id ?? "-"] ??= []).push(c);
  return groups;
}

export const CATALOG: { name: string; file: string; size: string }[] = [
  { name: "StyleTTS2 · LibriTTS", file: "styletts2_libritts.pth", size: "1.2 GB" },
  { name: "StyleTTS2 · LJSpeech", file: "styletts2_ljspeech.pth", size: "842 MB" },
  { name: "PL-BERT · multilingual", file: "plbert_ml.t7", size: "420 MB" },
  { name: "ASR · base aligner", file: "asr_base.pth", size: "310 MB" },
];
