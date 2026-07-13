# High-Volume Model Batching Implementation Plan

**Goal:** Move expensive model calls out of per-input node loops for high-volume audio, alignment, and synthesis families while preserving ordered outputs and fan-out.

**Architecture:** Nodes collect typed requests for the scheduler batch and call one collection-shaped adapter. Adapters use native model batching where available; singular-only libraries keep their bounded iteration behind the adapter so lifecycle, cancellation, and output validation remain centralized.

**Tech Stack:** runflow nodes, NeMo ASR, OpenAI Whisper, WhisperX, project TTS engine runtimes, StyleTTS2.

---

### Task 1: Batch ASR node model calls

**Files:** `src/runner/nodes/asr/nodes.py`, `src/runner/nodes/asr/whisper.py`

- Add collection-shaped Whisper transcription.
- Materialize every audio path once per node batch.
- Call Parakeet/Canary native multi-path APIs once and validate output cardinality.
- Rebuild transcript-bearing audio in input order and clean all temporary files reliably.

### Task 2: Batch WhisperX alignment calls

**Files:** `src/runner/nodes/asr/align.py`, `src/runner/nodes/asr/whisperx.py`

- Add a typed collection alignment adapter.
- Materialize the batch together and invoke the adapter once.
- Keep singular WhisperX calls internal because WhisperX exposes no multi-audio align API.
- Restore aligned segments in source order.

### Task 3: Add collection synthesis to TTS runtimes

**Files:** `src/runner/nodes/tts/engines/base.py`, `src/runner/nodes/tts/synthesis.py`

- Add typed synthesis requests/results and `synthesize_batch` to the engine boundary.
- Expand all text/voice/sample fan-out before calling the runtime.
- Invoke the runtime once per node batch and reconstruct ordered audio/results.
- Let engines override the default adapter when their library has a native batch API.

### Task 4: Batch StyleTTS request preparation and execution

**Files:** `src/runner/nodes/synthesis/styletts.py`, `src/runner/nodes/synthesis/styletts_runtime/actions.py`

- Expand all input/reference/sample combinations before synthesis.
- Bulk-read referenced audio IDs once and reuse one loaded runtime.
- Run through one collection adapter and preserve deterministic request IDs/output names.

### Task 5: Verify model boundaries and live graphs

- Use temporary static/adapter probes to prove nodes make collection calls.
- Compile all touched modules through Nix.
- Run existing ASR/TTS smoke workflows where required checkpoints are locally available.
- Inspect runs with the CLI and retain no temporary tests or graph files.
