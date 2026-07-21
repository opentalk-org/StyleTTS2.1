# LoadAudio Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `LoadAudio` return WAV bytes whose sample rate and channel count match its configured output metadata.

**Architecture:** Add a focused WAV conversion module to the `audio_io` node family and call it for every payload loaded by `LoadAudio`. Decode with SoundFile, transform channels explicitly, resample with librosa only when necessary, encode PCM-16 WAV, and derive output facts from the conversion result.

**Tech Stack:** Python, NumPy, SoundFile, librosa, Pydantic, runflow graph runtime

## Global Constraints

- Run all Python and test commands through `nix develop --command ...`.
- Validate node behavior through a real registered graph rather than calling `execute()` directly.
- Keep regression tests temporary and remove them before completion.
- Do not modify stored source audio bytes.

---

### Task 1: Normalize encoded WAV payloads

**Files:**
- Create: `src/runner/nodes/audio_io/conversion.py`
- Modify: `src/runner/nodes/audio_io/nodes.py`
- Temporary test: `/tmp/runflow_test_load_audio_conversion.py`

**Interfaces:**
- Produces: `normalize_wav_bytes(data: bytes, sample_rate: int, channels: int) -> ConvertedAudio`
- `ConvertedAudio` exposes `data: bytes`, `sample_rate: int`, `channels: int`, and `duration: float`.

- [ ] **Step 1: Write the failing conversion test**

Create a temporary test that generates a 48 kHz stereo WAV with distinct left/right samples, requests 24 kHz mono conversion, decodes the result, and asserts one channel, 24 kHz, stable duration, and averaged channel content. Add cases proving mono expansion duplicates samples and multichannel reduction/expansion follows the approved deterministic rules.

- [ ] **Step 2: Run the test to verify it fails**

Run: `nix develop --command python -m pytest /tmp/runflow_test_load_audio_conversion.py -v`

Expected: collection fails because `runner.nodes.audio_io.conversion` does not exist.

- [ ] **Step 3: Implement minimal conversion**

Define an immutable `ConvertedAudio` record. Decode with `soundfile.read(..., always_2d=True, dtype="float32")`, reject zero frames, average all channels for mono, duplicate mono for expansion, truncate for non-mono reduction, repeat the last channel for non-mono expansion, resample each channel with `librosa.resample`, and encode an always-two-dimensional PCM-16 WAV.

Update `LoadAudioNode.execute` to normalize every resolved payload and replace `data`, `sample_rate`, `channels`, `end`, and `byte_length`. Update annotation metadata with normalized `duration`, `sample_rate`, `channels`, `byte_length`, and the original `source_duration`.

- [ ] **Step 4: Run focused checks**

Run: `nix develop --command python -m pytest /tmp/runflow_test_load_audio_conversion.py -v`

Expected: all temporary conversion tests pass.

Run: `nix develop --command python -m compileall -q src/runner/nodes/audio_io`

Expected: exit code 0.

- [ ] **Step 5: Validate through a registered graph**

Create a temporary graph using the repository's testing feeder, `LoadAudio`, and an output node; submit it through `POST /graphs/runs`, then inspect it with `nix develop --command python -m cli logs <run_id>` and verify the emitted WAV header and `Audio` metadata agree. If the shared stack is unavailable, report that exact environmental limitation and retain the focused automated evidence.

- [ ] **Step 6: Clean temporary artifacts and review**

Remove the temporary test and graph files, run `git diff --check`, inspect the scoped diff, and verify no unrelated user changes were altered.
