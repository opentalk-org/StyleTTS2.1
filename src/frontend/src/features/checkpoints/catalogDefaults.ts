import type { CatalogItem } from "./logic";

export const CORE_CATALOG_ITEMS: CatalogItem[] = [
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
];
