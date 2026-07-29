import type { Tone } from "@/shared/ui/Badge";
import type { Checkpoint } from "./api";

export const TYPE_TONE: Record<string, Tone> = {
  kokoro: "red",
  chatterbox: "red",
  f5_tts: "red",
  orpheus: "red",
  dia: "red",
  fish_speech: "red",
  piper: "red",
  styletts2: "blue",
  asr: "emerald",
  f0: "amber",
  plbert: "gray",
  whisper: "emerald",
  parakeet: "emerald",
  canary: "emerald",
  whisperx: "blue",
  sortformer: "amber",
  mos_base: "blue",
  mos_model: "emerald",
};

export function checkpointTone(type: string): Tone {
  return TYPE_TONE[type] ?? "gray";
}

export function checkpointSymbolCount(checkpoint: Checkpoint | undefined): number | null {
  if (!checkpoint) return null;
  const raw = checkpoint.metadata.symbols;
  if (Array.isArray(raw)) return raw.length;
  const count = Number(raw);
  return Number.isFinite(count) && count > 0 ? count : null;
}

export function checkpointSymbols(checkpoint: Checkpoint | undefined): string[] {
  if (!checkpoint) return [];
  const raw = checkpoint.metadata.symbols;
  return Array.isArray(raw) ? raw.map((symbol) => String(symbol)) : [];
}

export function checkpointDecoderType(checkpoint: Checkpoint | undefined): string {
  if (!checkpoint) return "";
  return String(checkpoint.metadata.decoder_type ?? "").trim().toLowerCase();
}

export function checkpointMultispeaker(checkpoint: Checkpoint | undefined): boolean | null {
  if (!checkpoint) return null;
  const raw = checkpoint.metadata.multispeaker;
  if (typeof raw === "boolean") return raw;
  return null;
}

export function checkpointSpeakerMode(checkpoint: Checkpoint | undefined): string {
  const multispeaker = checkpointMultispeaker(checkpoint);
  if (multispeaker === null) return "";
  return multispeaker ? "multi" : "single";
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
