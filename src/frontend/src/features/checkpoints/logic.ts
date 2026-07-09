import type { Tone } from "@/shared/ui/Badge";
import type { Checkpoint } from "./api";

export const TYPE_TONE: Record<string, Tone> = {
  kokoro: "red",
  chatterbox: "red",
  f5_tts: "red",
  orpheus: "red",
  dia: "red",
  fish_speech: "red",
  styletts2: "blue",
  asr: "emerald",
  f0: "amber",
  plbert: "gray",
  whisper: "emerald",
  parakeet: "emerald",
  canary: "emerald",
  whisperx: "blue",
  sortformer: "amber",
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
  return String(checkpoint.metadata.decoder_type ?? checkpoint.metadata.decoder ?? "").trim().toLowerCase();
}

export function checkpointMultispeaker(checkpoint: Checkpoint | undefined): boolean | null {
  if (!checkpoint) return null;
  const raw = checkpoint.metadata.multispeaker;
  if (typeof raw === "boolean") return raw;
  if (typeof raw === "string") {
    const normalized = raw.trim().toLowerCase();
    if (["true", "multi", "multispeaker"].includes(normalized)) return true;
    if (["false", "single", "single_speaker"].includes(normalized)) return false;
  }
  const legacy = checkpoint.metadata.spkMode ?? checkpoint.metadata.speaker_mode;
  if (typeof legacy === "string") {
    const normalized = legacy.trim().toLowerCase();
    if (normalized === "multi") return true;
    if (normalized === "single") return false;
  }
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

export type CatalogItem = {
  name: string;
  file: string;
  group: "TTS" | "StyleTTS2" | "Training assets" | "Transcription" | "Diarization";
  catalogKey: string;
  item: string;
};

export function groupCatalogItems(items: CatalogItem[]): Record<CatalogItem["group"], CatalogItem[]> {
  const groups: Record<CatalogItem["group"], CatalogItem[]> = {
    TTS: [],
    StyleTTS2: [],
    "Training assets": [],
    Transcription: [],
    Diarization: [],
  };
  for (const item of items) groups[item.group].push(item);
  return groups;
}

export const CATALOG: CatalogItem[] = [
  {
    name: "Kokoro · 82M (8 langs, 54 voices)",
    file: "hexgrad/Kokoro-82M",
    group: "TTS",
    catalogKey: "tts_models",
    item: "kokoro",
  },
  {
    name: "Chatterbox · multilingual (~23 langs, clone)",
    file: "ResembleAI/chatterbox",
    group: "TTS",
    catalogKey: "tts_models",
    item: "chatterbox",
  },
  {
    name: "F5-TTS · v1 base (EN/ZH, clone)",
    file: "SWivid/F5-TTS",
    group: "TTS",
    catalogKey: "tts_models",
    item: "f5_tts",
  },
  {
    name: "Orpheus · 3B (EN, 8 voices)",
    file: "unsloth/orpheus-3b-0.1-ft",
    group: "TTS",
    catalogKey: "tts_models",
    item: "orpheus",
  },
  {
    name: "Dia · 1.6B (EN dialogue, clone)",
    file: "nari-labs/Dia-1.6B-0626",
    group: "TTS",
    catalogKey: "tts_models",
    item: "dia",
  },
  {
    name: "Fish S2-Pro · dual-AR (80+ langs, clone)",
    file: "fishaudio/s2-pro",
    group: "TTS",
    catalogKey: "tts_models",
    item: "fish_speech",
  },
  {
    name: "StyleTTS2 · LibriTTS",
    file: "epochs_2nd_00020.pth",
    group: "StyleTTS2",
    catalogKey: "official_checkpoints",
    item: "official_styletts2_libritts",
  },
  {
    name: "StyleTTS2 · LJSpeech",
    file: "epoch_2nd_00100.pth",
    group: "StyleTTS2",
    catalogKey: "official_checkpoints",
    item: "official_styletts2_ljspeech",
  },
  {
    name: "StyleTTS2 · Vokan",
    file: "epoch_2nd_00012.pth",
    group: "StyleTTS2",
    catalogKey: "vokan_checkpoint",
    item: "vokan_styletts2",
  },
  {
    name: "PL-BERT · multilingual",
    file: "step_1100000.t7",
    group: "Training assets",
    catalogKey: "papercup_multilingual_pl_bert",
    item: "papercup_multilingual_pl_bert",
  },
  {
    name: "ASR · base aligner",
    file: "epoch_00080.pth",
    group: "Training assets",
    catalogKey: "styletts2_utils",
    item: "styletts2_utils_asr",
  },
  {
    name: "F0 · JDC",
    file: "bst.t7",
    group: "Training assets",
    catalogKey: "styletts2_utils",
    item: "styletts2_utils_f0",
  },
  {
    name: "PL-BERT · StyleTTS2 utils",
    file: "step_1000000.t7",
    group: "Training assets",
    catalogKey: "styletts2_utils",
    item: "styletts2_utils_plbert",
  },
  {
    name: "Whisper · tiny",
    file: "tiny.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:tiny",
  },
  {
    name: "Whisper · tiny.en",
    file: "tiny.en.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:tiny.en",
  },
  {
    name: "Whisper · base",
    file: "base.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:base",
  },
  {
    name: "Whisper · base.en",
    file: "base.en.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:base.en",
  },
  {
    name: "Whisper · small",
    file: "small.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:small",
  },
  {
    name: "Whisper · small.en",
    file: "small.en.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:small.en",
  },
  {
    name: "Whisper · medium",
    file: "medium.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:medium",
  },
  {
    name: "Whisper · medium.en",
    file: "medium.en.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:medium.en",
  },
  {
    name: "Whisper · large",
    file: "large.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:large",
  },
  {
    name: "Whisper · large-v1",
    file: "large-v1.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:large-v1",
  },
  {
    name: "Whisper · large-v2",
    file: "large-v2.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:large-v2",
  },
  {
    name: "Whisper · large-v3",
    file: "large-v3.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:large-v3",
  },
  {
    name: "Whisper · turbo",
    file: "large-v3-turbo.pt",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisper:turbo",
  },
  {
    name: "Parakeet · TDT 0.6B v2",
    file: "parakeet-tdt-0.6b-v2.nemo",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "parakeet:nvidia/parakeet-tdt-0.6b-v2",
  },
  {
    name: "Parakeet · TDT 0.6B v3",
    file: "parakeet-tdt-0.6b-v3.nemo",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "parakeet:nvidia/parakeet-tdt-0.6b-v3",
  },
  {
    name: "Canary · 1B v2",
    file: "canary-1b-v2.nemo",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "canary:nvidia/canary-1b-v2",
  },
  {
    name: "Canary · 1B",
    file: "canary-1b.nemo",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "canary:nvidia/canary-1b",
  },
  {
    name: "Canary · 1B Flash",
    file: "canary-1b-flash.nemo",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "canary:nvidia/canary-1b-flash",
  },
  {
    name: "Canary · 180M Flash",
    file: "canary-180m-flash.nemo",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "canary:nvidia/canary-180m-flash",
  },
  {
    name: "Sortformer · 4spk v1",
    file: "diar_sortformer_4spk-v1.nemo",
    group: "Diarization",
    catalogKey: "asr_models",
    item: "sortformer:nvidia/diar_sortformer_4spk-v1",
  },
  {
    name: "WhisperX align · English",
    file: "wav2vec2-base-960h",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisperx:facebook/wav2vec2-base-960h",
  },
  {
    name: "WhisperX align · Polish",
    file: "wav2vec2-large-xlsr-53-polish",
    group: "Transcription",
    catalogKey: "asr_models",
    item: "whisperx:jonatasgrosman/wav2vec2-large-xlsr-53-polish",
  },
];
