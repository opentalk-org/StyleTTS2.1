# TTS engine capabilities (research table — source of truth)

Researched 2026-07 against HuggingFace model cards and GitHub READMEs. This table
drives which nodes exist, which support voice cloning, which have preset voices,
and what has to be downloaded / installed.

Shared runtime constraint: this repo pins **torch 2.11 (cu128), transformers 4.57.x,
numpy 2.x** for the RTX 5090 (Blackwell / sm_120). Rather than isolate the engines,
the base env **forces the shared stack up** via `[tool.uv] override-dependencies`
(torch/torchaudio/torchvision, numpy, transformers<5, nemo-toolkit>=2.7, onnx>=1.19)
so the engine wheels resolve against the newer stack instead of dragging it down.
All engine runtimes still **lazy-import** their library, so a missing/unusable engine
raises a clear `*_not_installed` error instead of breaking node discovery.

## In-environment status (torch 2.11, forced versions)

| Engine | In base env? | Verified | Notes |
|---|---|---|---|
| Kokoro | ✅ installed (`kokoro`) | real audio, 2 voices, 24 kHz | needs `en-core-web-sm` (shipped) + OS `espeak-ng` |
| Chatterbox | ✅ installed (`chatterbox-tts`) | real audio, 24 kHz | multilingual + English models load and synthesize on torch 2.11 |
| Dia | ✅ available (no extra pkg) | `DiaForConditionalGeneration` imports | transformers 4.57.6 ≥ 4.53; only the model download is needed |
| F5-TTS | ✅ installed (`f5-tts`) | real audio (cloning), 24 kHz | needs `torchcodec==0.11.1` (torch-2.11 ABI) + ffmpeg-7 libs on `LD_LIBRARY_PATH` (added in `flake.nix`) |
| Orpheus | ✅ installed (`orpheus-speech`+vLLM) | real audio, 24 kHz (voice "tara") | vLLM 0.21 **cu129** wheel (pinned in `[tool.uv.sources]`) links `libcudart.so.12` → runs on cu128 torch. Runtime drives vLLM's offline `LLM` API directly + SNAC decode. Uses the ungated `unsloth/orpheus-3b-0.1-ft` mirror; runs the engine in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) with `VLLM_USE_FLASHINFER_SAMPLER=0` + FLASH_ATTN (flashinfer's JIT kernel links libcudart.so.13) |
| Fish Speech | ❌ not force-installable | — | PyPI `fish-speech` 0.1.0 pins `datasets==2.18.0`, whose `fsspec` conflicts with nemo-toolkit>=2.7; forcing it would break the ASR stack |
| Raon-OpenTTS | ❌ no wheel | — | git-only F5 fork (`pip install -e .`); shares F5's torchcodec limitation |

Kokoro, Chatterbox, F5-TTS are in `pyproject.toml` base dependencies. Orpheus, Fish
Speech, and Raon are not force-installed because doing so either can't run on torch
2.11 (vLLM) or would downgrade the ASR stack (fsspec/datasets); their nodes remain
importable and raise `*_not_installed` until installed some other way.

## Summary table

| Engine | HF repo | #Voices | Languages | Voice cloning | pip package | Inference entrypoint | License | Sample rate |
|---|---|---|---|---|---|---|---|---|
| Kokoro | `hexgrad/Kokoro-82M` | 54 | 8 (EN×2, ES, FR, HI, IT, JA, PT-BR, ZH) | **NO** (fixed embeddings) | `kokoro` | `from kokoro import KPipeline` | Apache-2.0 | 24 kHz |
| Chatterbox | `ResembleAI/chatterbox` | 0 (zero-shot) | ~23 multilingual; EN-only variant | **YES** — ref audio ~10s, no transcript | `chatterbox-tts` | `from chatterbox.tts import ChatterboxTTS` | MIT | 24 kHz |
| F5-TTS | `SWivid/F5-TTS` | 0 (zero-shot) | EN + ZH (community FTs add more) | **YES** — ref audio + transcript | `f5-tts` | `from f5_tts.api import F5TTS` | code MIT / weights CC-BY-NC-4.0 | 24 kHz |
| Orpheus | `canopylabs/orpheus-3b-0.1-ft` | 8 (tara, leah, jess, leo, dan, mia, zac, zoe) | EN (+7-lang preview) | **YES** — prompt example pairs (zero-shot) | `orpheus-speech` (vLLM) | `from orpheus_tts import OrpheusModel` | Apache-2.0 | 24 kHz |
| Dia / Dia2 | `nari-labs/Dia-1.6B-0626` (+ `Dia2-1B`/`-2B`) | 0 (no fixed speaker) | EN only | **YES** — audio prompt + transcript | HF Transformers (`DiaForConditionalGeneration`) | `transformers` ≥4.53 | Apache-2.0 | Dia1 44.1 kHz; Dia2 ~24 kHz |
| Fish Speech / OpenAudio S1 | `fishaudio/openaudio-s1-mini`, `fishaudio/fish-speech-1.5` | 0 (clone-based) | 13 | **YES** — ref audio + transcript | `fish-speech` (source) | CLI `text2semantic` / HTTP API | CC-BY-NC-SA-4.0 | 44.1 kHz |
| Raon-OpenTTS | `KRAFTON/Raon-OpenTTS-0.3B`/`-1B` | 0 (zero-shot) | EN only | **YES** — ref audio + transcript | `-e .` (F5-TTS fork) | `from f5_tts.infer.utils_infer import infer_process` | Apache-2.0 (repo) vs CC-BY-NC-4.0 (weights) | 16 kHz |

## Which nodes exist (derived from the table)

- **Synthesis node (text + voice → audio): every engine.** `KokoroSynthesis`,
  `ChatterboxSynthesis`, `F5TtsSynthesis`, `OrpheusSynthesis`, `DiaSynthesis`,
  `FishSpeechSynthesis`, `RaonOpenTtsSynthesis`.
- **Preset-voice selection: engines with presets only** — Kokoro (54), Orpheus (8).
  Exposed through the generic `TtsSelectVoice` and `TtsRandomVoices` nodes (engine
  is a setting; preset lists live in `tts/voices.py`).
- **Voice cloning node: every engine that supports it** — Chatterbox, F5-TTS,
  Orpheus, Dia, Fish Speech, Raon-OpenTTS. `*CloneVoice` nodes take a reference
  `audio` (load it with the existing `LoadAudio`/`AudioSource` nodes) plus an
  optional `transcript`. Kokoro has **no** clone node (fixed embeddings).
- **Loading reference audio** reuses the existing generic `LoadAudio` node — no
  per-engine loader is added (see AGENTS.md "reuse ports / single generalized").

## Install / dependency notes

- Base `pyproject.toml` dependencies gain `kokoro`, `chatterbox-tts`, `f5-tts`, and
  `en-core-web-sm` (Kokoro's English G2P model, pinned by URL so it is never fetched
  at runtime). Kokoro also needs the OS package `espeak-ng` (provided by the flake).
- `[tool.uv] override-dependencies` forces `transformers>=4.45,<5`, `nemo-toolkit>=2.7`,
  and `onnx>=1.19` (on top of the existing torch/numpy overrides) so adding kokoro does
  not backtrack nemo to 2.1.0 / onnx 1.12.0 (an unbuildable combo that breaks `uv sync`).
- Dia runs through `transformers` (`DiaForConditionalGeneration`) — already available at
  transformers 4.57.6; no extra PyPI package, just the model download.
- Orpheus (`orpheus-speech`, vLLM/torch-2.5) and Fish Speech (`fish-speech` PyPI stub,
  `datasets==2.18` vs nemo `fsspec`) are intentionally NOT in base deps: they cannot run
  on torch 2.11 / would break the ASR stack. Raon-OpenTTS has no wheel (git F5 fork).

## Flagged uncertainties

- Chatterbox exact language count (21/23/25) and dependency specifiers.
- F5-TTS "13 languages" claim (base is EN/ZH only; others are community fine-tunes).
- Orpheus smaller variants (0.5B/1B/150M) appear to be roadmap, not released.
- Dia2 output sample rate not documented (assumed ~24 kHz via Mimi codec).
- Fish Speech 1.5 param count unstated; 44.1 kHz from docs, not the HF card.
- Raon-OpenTTS license contradicts between GitHub (Apache-2.0) and HF weights (CC-BY-NC-4.0).
</content>
