# FineWiki Other-TTS Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize the 96 registered FineWiki streams through every compatible non-Piper/non-Kokoro TTS engine, using matching stored TTS outputs as clone references and preserving all existing audio.

**Architecture:** A dataset CRUD query exposes lightweight reference candidates, and a corpus reference service deterministically selects and bulk-loads one clip per registered stream. An engine-generic corpus node builds text-sensitive jobs, keeps one checkpoint runtime loaded, resumes from dataset source keys, and emits transcript-bearing audio to the existing atomic writer. A launcher downloads checkpoints and submits one accelerator-exclusive engine graph at a time.

**Tech Stack:** Python 3.12, Runflow, SQLAlchemy/PostgreSQL, packed audio CRUD, PyTorch/vLLM, existing TTS engine adapters, FastAPI graph API.

## Global Constraints

- Run Python, tests, backend, and CLI only through `nix develop --command`.
- Test nodes through `POST /graphs/runs`; never call `execute()` directly.
- Preserve all existing audio and dataset memberships.
- Use the 96 existing `registered-*` streams and their 450-line TXT files.
- Select clone references only from `tts_piper` and `tts_kokoro`.
- Reject references outside 4–12 seconds or without one nonempty full-duration transcript.
- Source keys include normalized-text digest and reference audio ID.
- Store outputs in `tts_<engine>` with matching `tts_dataset` metadata.
- Keep files below 300 lines and folders below 16 files.
- Temporary tests live under `/tmp` and are removed after verification.

---

### Task 1: Deterministic Stored-TTS Reference Selection

**Files:**
- Modify: `src/shared/db/datasets/crud.py`
- Create: `src/runner/nodes/tts/corpus/references.py`
- Test: `/tmp/test_tts_corpus_references.py`

**Interfaces:**
- Produces: `list_tts_reference_candidates(session, dataset_ids, streams) -> Sequence[AudioFile]`.
- Produces: `RegisteredReference(stream_id, language, audio_file_id, transcript, wav_bytes)`.
- Produces: `load_registered_references(dataset_ids, stream_languages) -> Mapping[str, RegisteredReference]`.

- [ ] **Step 1: Write the failing reference-selection test**

Create fixtures for two registered streams with candidates before, inside, and
after the 4–12 second interval. Assert selection minimizes
`(abs(duration - 8), sentence_index, str(audio_id))`, rejects a mismatched
language or segment span, and bulk-loads bytes only for the two winners.

```python
selected = select_reference_rows(rows, {"registered-en-000": "en"})
assert selected["registered-en-000"].duration == 8.25
assert selected["registered-en-000"].segments[0]["text"] == "reference text"
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```bash
nix develop --command env PYTHONPATH=src python -m unittest /tmp/test_tts_corpus_references.py -v
```

Expected: import failure for `runner.nodes.tts.corpus.references`.

- [ ] **Step 3: Add the bounded dataset CRUD query**

Implement `list_tts_reference_candidates` with one SQLAlchemy statement joining
`dataset_audio_files`, filtering the provided dataset UUIDs and
`AudioFile.metadata_["stream"].astext.in_(streams)`, ordering by stream and
audio UUID. Return metadata rows only; do not read pack files in this function.

- [ ] **Step 4: Implement selection and bulk byte loading**

Define immutable `RegisteredReference` and a pure `select_reference_rows`.
Require stored, nonvirtual audio, matching metadata language, one segment with
`start == 0`, `end == duration`, and nonempty text. Use
`audio_crud.bulk_read_audio_files` once for selected UUIDs, then construct the
reference mapping. Raise `ValueError("tts_reference_missing:<stream>")` for
every missing stream.

- [ ] **Step 5: Verify green and commit**

Run the focused test, then:

```bash
git add src/shared/db/datasets/crud.py src/runner/nodes/tts/corpus/references.py
git commit -m "feat: select stored TTS clone references"
```

### Task 2: Other-Engine Corpus Planning

**Files:**
- Modify: `src/runner/nodes/tts/corpus/models.py`
- Create: `src/runner/nodes/tts/corpus/other_plan.py`
- Test: `/tmp/test_other_tts_corpus_plan.py`

**Interfaces:**
- Produces: `OtherCorpusJob(engine, stream_id, language, sentence_index, text, voice_id, reference_audio_id, source_key)`.
- Produces: `build_other_corpus_plan(root, engine, references) -> tuple[OtherCorpusJob, ...]`.
- Produces exact job totals: Chatterbox 43,200; F5-TTS 17,100; Orpheus 3,600; Dia 14,850; Fish Speech 43,200; Raon OpenTTS 14,850.

- [ ] **Step 1: Write failing plan-contract tests**

Load the real manifest, use UUID-only fake references for all 96 registered
streams, and assert exact per-engine totals and supported languages. Assert
Orpheus produces 450 jobs for each of its eight presets and clone engines bind
each job to its stream reference.

```python
assert len(build_other_corpus_plan(root, TtsEngine.CHATTERBOX, refs)) == 43_200
assert len(build_other_corpus_plan(root, TtsEngine.ORPHEUS, refs)) == 3_600
```

Also change one line while retaining stream/index and assert its source key
changes.

- [ ] **Step 2: Run the focused test and verify red**

Run:

```bash
nix develop --command env PYTHONPATH=src python -m unittest /tmp/test_other_tts_corpus_plan.py -v
```

Expected: import failure for `runner.nodes.tts.corpus.other_plan`.

- [ ] **Step 3: Implement explicit engine capabilities**

Define immutable language sets:

```python
ENGINE_LANGUAGES = {
    TtsEngine.CHATTERBOX: frozenset({"en","de","fr","nl","zh","ja","hi","es","pt","it","ru","pl","ar","tr","ko"}),
    TtsEngine.F5_TTS: frozenset({"en", "zh"}),
    TtsEngine.DIA: frozenset({"en"}),
    TtsEngine.FISH_SPEECH: frozenset({"en","de","fr","nl","zh","ja","hi","es","pt","it","ru","pl","ar","tr","ko"}),
    TtsEngine.RAON_OPENTTS: frozenset({"en"}),
}
```

Keep Orpheus separate because its eight preset voices consume eight English
registered TXT files but do not use clone references.

- [ ] **Step 4: Implement text-sensitive job construction**

Read only `kind == "registered"` manifest records, validate TXT counts, and
compute a 12-byte Blake2 digest of normalized text. Clone keys are:

```python
f"{engine.value}:{stream}:{reference_id}:{index:04d}:{digest}"
```

Orpheus keys replace `stream:reference_id` with the preset voice ID. Validate
unique keys and the exact configured total before returning.

- [ ] **Step 5: Verify green and commit**

Run the focused test twice, compare serialized keys, then:

```bash
git add src/runner/nodes/tts/corpus/models.py src/runner/nodes/tts/corpus/other_plan.py
git commit -m "feat: plan remaining TTS corpus engines"
```

### Task 3: Resumable Generic Other-TTS Corpus Node

**Files:**
- Create: `src/runner/nodes/tts/corpus/other.py`
- Modify: `src/runner/nodes/tts/corpus/audio.py`
- Modify: `src/runner/nodes/tts/corpus/__init__.py`
- Test: `/tmp/test_other_tts_corpus_node.py`

**Interfaces:**
- Produces registered node type `OtherTtsCorpusSynthesis`.
- Settings: `engine`, `corpus_dir`, `source_dataset_ids`, `dataset_id`, `dataset_name`, `checkpoint_id`, `batch_size`, and `max_jobs`.
- Emits `audio: AudioPort(mode=STREAM)` with one full-duration transcript.

- [ ] **Step 1: Write failing schema and metadata tests**

Assert the registry exports the node, its engine enum excludes Piper and
Kokoro, its output is streaming, and its resource policy is accelerator
exclusive. Test the pure audio builder with an `OtherCorpusJob` and assert
dataset flag, reference ID, text-sensitive key, speaker/stream identity, and
one transcript segment spanning the complete duration.

- [ ] **Step 2: Run focused tests and verify red**

Run:

```bash
nix develop --command env PYTHONPATH=src python -m unittest /tmp/test_other_tts_corpus_node.py -v
```

Expected: missing `OtherTtsCorpusSynthesisNode`.

- [ ] **Step 3: Implement setup and resume**

In `setup`, validate `dataset_name == f"tts_{engine.value}"`, resolve the
checkpoint through `resolve_checkpoint_ref`, load references from the two
configured source datasets, build the engine plan, filter
`completed_source_keys`, and load one engine runtime. Apply `max_jobs` before
resume filtering so smoke runs remain deterministic.

- [ ] **Step 4: Implement bounded synthesis**

For clone jobs, construct `Voice(engine, None, CloneReference(...))`. For
Orpheus, construct `Voice(engine, job.voice_id, None)`. Call the runtime's
`synthesize_batch` through `asyncio.to_thread`, check cancellation between
batches, report item progress, and emit one output per result. Use a static
exclusive resource policy of one accelerator and 30 GiB VRAM to prevent
concurrent large-model residency.

- [ ] **Step 5: Register and verify green**

Add the node to `TTS_CORPUS_NODES`, run the focused test, restart the shared
session, and verify `/schema` contains the exact settings and stream port.

- [ ] **Step 6: Commit**

```bash
git add src/runner/nodes/tts/corpus
git commit -m "feat: synthesize resumable other TTS corpora"
```

### Task 4: Idempotent Launcher and Real-Graph Smoke Tests

**Files:**
- Create: `imports/run_other_tts_corpus_campaign.py`
- Test: `/tmp/test_other_tts_campaign_launcher.py`

**Interfaces:**
- Produces CLI options `--engine`, `--smoke`, `--max-jobs`, and `--backend-url`.
- Creates or reuses datasets `tts_chatterbox`, `tts_f5_tts`, `tts_orpheus`, `tts_dia`, `tts_fish_speech`, and `tts_raon_opentts`.
- Downloads the matching catalog checkpoint before synthesis.

- [ ] **Step 1: Write the failing request-construction test**

Assert each engine request contains one `OtherTtsCorpusSynthesis` node, one
`SaveAudioRecord` node with the matching dataset UUID, one audio edge, and
runtime resources `accelerator=1`, `vram_gb=30`, and `io=4`.

- [ ] **Step 2: Verify red**

Run:

```bash
nix develop --command env PYTHONPATH=src python -m unittest /tmp/test_other_tts_campaign_launcher.py -v
```

Expected: import failure for `imports.run_other_tts_corpus_campaign`.

- [ ] **Step 3: Implement checkpoint/dataset preparation**

Reuse the existing backend JSON request pattern. Resolve checkpoint by exact
`type_`; if absent, submit `CatalogDownload(catalog_key="tts_models",
item=engine.value)` and wait for success. Resolve or create the exact
`tts_<engine>` dataset without deleting anything.

- [ ] **Step 4: Implement sequential engine submission**

Default engine order is Chatterbox, F5-TTS, Orpheus, Dia, Fish Speech, then
Raon OpenTTS. `--smoke` sets `max_jobs=1`; full mode leaves it unset. Wait for
each graph to reach a terminal state before submitting the next so accelerator
models never overlap.

- [ ] **Step 5: Verify green and run all smoke graphs**

Run the launcher test, then:

```bash
nix develop --command python imports/run_other_tts_corpus_campaign.py --smoke
nix develop --command python -m cli runs
```

For a failure, inspect `python -m cli failed <run_id>`, fix the owning engine
adapter with a focused failing test, and rerun that engine's smoke graph.

- [ ] **Step 6: Commit**

```bash
git add imports/run_other_tts_corpus_campaign.py
git commit -m "feat: launch remaining TTS corpus engines"
```

### Task 5: Full Campaign and Database Audit

**Files:**
- Test: `/tmp/audit_other_tts_campaign.py`

**Interfaces:**
- Produces six resumable dataset runs totaling 136,800 planned source keys.
- Proves dataset/source-key equality and transcript completeness.

- [ ] **Step 1: Capture preservation baseline**

Using shared CRUD facades, record total audio count and all dataset membership
counts outside the six target datasets in `/tmp/other_tts_baseline.json`.

- [ ] **Step 2: Launch the full sequential campaign**

Run:

```bash
nix develop --command python imports/run_other_tts_corpus_campaign.py
```

Monitor terminal states with `python -m cli runs` and inspect failures with
`python -m cli failed <run_id>`. Resubmit the same engine after a fix; resume
must skip committed source keys.

- [ ] **Step 3: Audit each completed dataset**

Build the real plans again and assert dataset metadata source keys equal the
planned set. For every member assert positive bytes/duration, matching
`tts_dataset`, engine/reference metadata, and exactly one segment with
`start == 0`, `end == duration`, and the planned transcript.

- [ ] **Step 4: Prove preservation and run final checks**

Assert every non-target membership count is at least its baseline and total
audio increased by exactly the sum of new unique target source keys. Run:

```bash
nix develop --command python -m compileall -q src/runner/nodes/tts imports/run_other_tts_corpus_campaign.py
git diff --check
```

- [ ] **Step 5: Remove temporary tests**

Move only the temporary campaign test files and baseline to trash:

```bash
gio trash /tmp/test_tts_corpus_references.py /tmp/test_other_tts_corpus_plan.py /tmp/test_other_tts_corpus_node.py /tmp/test_other_tts_campaign_launcher.py /tmp/audit_other_tts_campaign.py /tmp/other_tts_baseline.json
```
