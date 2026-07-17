# Beetle Database Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a database-native, bulk-prefetched, continuously cycling data pipeline that returns every tensor and mask required by all three Beetle stages.

**Architecture:** Cursor-paged shared CRUD builds a compact immutable segment index. Deterministic planners select sentence/mid-sentence targets and grouped style/voice views; a bounded source resolves current JSONB segments and all WAV ranges in bulk before typed collation.

**Tech Stack:** Python dataclasses/Pydantic, NumPy, PyTorch, torchaudio, shared SQLAlchemy CRUD, packed-audio range reader, Nix.

## Global Constraints

- PostgreSQL metadata and shared audio CRUD/range helpers remain authoritative.
- `voice_id` is canonical; never fall back to free-form `speaker`.
- Accept target durations from 1 through 45 seconds.
- Keep context availability separate from model-side dropout.
- Do not materialize full datasets or duplicate complete JSONB segments in memory.
- Use bulk metadata/audio operations and bounded prefetch measured in batches and decoded bytes.
- Use temporary tests under `/tmp`; no committed tests.

---

### Task 1: Typed stored-record and batch contracts

**Files:**
- Create: `src/runner/nodes/training/beetle/data/__init__.py`
- Create: `src/runner/nodes/training/beetle/data/records.py`
- Create temporarily: `/tmp/test_beetle_data.py`

**Interfaces:**
- Produces: `SegmentKey`, `IndexedSegment`, `CutRange`, `ContextRanges`, `PlannedExample`, `DecodedExample`, and `BeetleBatch`.
- Consumes: UUIDs, tensors, and exact DB segment fields only.

- [ ] Write failing construction tests for immutable keys `(audio_file_id, segment_index, segment_id)`, 1–45 second cut invariants, optional file-level text prompts, distinct context-availability masks, and `BeetleBatch.to(device)` preserving integer/bool dtypes.
- [ ] Run `nix develop --command pytest -q /tmp/test_beetle_data.py`; expect import failure.
- [ ] Implement focused frozen dataclasses. `BeetleBatch` must expose waveform, mel, phoneme IDs, text IDs, durations/alignment, target/context/pair lengths and masks, voice IDs, prompt tokens, sample keys, and deterministic view seeds.
- [ ] Re-run tests; expect PASS.
- [ ] Commit: `git commit -m 'feat: define beetle data contracts'`.

### Task 2: Compact database segment index and eligibility

**Files:**
- Create: `src/runner/nodes/training/beetle/data/index.py`
- Modify temporarily: `/tmp/test_beetle_data.py`

**Interfaces:**
- Produces: `DatabaseSegmentIndex.build(selection, page_size, callbacks)`, `EligibilityReport`, `StagePools`, `index_fingerprint()`.
- Consumes: `audio_crud.count_segment_references`, `audio_crud.list_segment_references_page`, strict config selection, Task 1 records.

- [ ] Add a fake-CRUD test with multiple audio files, segment voices, prompts, virtual rows, missing phonemes, aligned/unaligned segments, and adjacent different voices. Assert stage-specific counts, chronological per-audio order, duration buckets, voice groups, sentence/mid-sentence pools, and stable fingerprint.
- [ ] Run focused tests; expect failure because `data.index` is absent.
- [ ] Implement cursor paging with cancellation/progress callbacks. Retain compact scalar/key metadata and per-audio prompt values, not complete segment JSON.
- [ ] Require stored, non-virtual audio. Stage 1 eligibility requires valid duration; Stages 2/3 additionally require nonempty text/phon and `voice_id`; mid-sentence eligibility additionally requires word alignment mapped to phoneme word groups.
- [ ] Re-run tests; expect exact eligibility counts and PASS.
- [ ] Commit: `git commit -m 'feat: index beetle training segments'`.

### Task 3: Sentence, mid-sentence, context, and pair planning

**Files:**
- Create: `src/runner/nodes/training/beetle/data/cuts.py`
- Create: `src/runner/nodes/training/beetle/data/sampling.py`
- Modify temporarily: `/tmp/test_beetle_data.py`

**Interfaces:**
- Produces: `CutPlanner.plan(key, seed)`, `ContinuousBatchPlanner.next_batch()`, `planner_state_dict()`, `load_planner_state_dict()`.
- Consumes: `StagePools`, indexed temporal neighbours, configured sentence ratio, duration buckets, and voice/style group sizes.

- [ ] Add failing deterministic tests: sentence cuts preserve full ranges; mid-sentence cuts use aligned word boundaries and matching phoneme groups; pre/post ranges use excluded or adjacent same-file audio; adjacent voices may differ; voice batches contain configured utterances per voice; style batches contain distance-weighted same-recording views plus negatives.
- [ ] Assert two planners restored at the same `cycle_index`, permutation, next-batch position, and seed produce identical planned examples indefinitely across a cycle boundary.
- [ ] Implement stateless random choices derived from stage, cycle, batch, sample key, and view ID. Maintain separate eligible pools so the requested sentence/mid-sentence ratio never silently falls back.
- [ ] Re-run tests; expect PASS with no epoch field or counter.
- [ ] Commit: `git commit -m 'feat: plan beetle training samples'`.

### Task 4: Bulk JSONB and packed-WAV prefetch

**Files:**
- Create: `src/runner/nodes/training/beetle/data/source.py`
- Create: `src/runner/nodes/training/beetle/data/audio.py`
- Modify temporarily: `/tmp/test_beetle_data.py`

**Interfaces:**
- Produces: `DatabaseBatchSource.fetch(plans)`, `AudioPreprocessor.decode(clip)`, and `BoundedBatchPrefetcher`.
- Consumes: `audio_crud.list_audio_segments_bulk`, `bulk_read_wav_segments`, `SegmentReadRequest`, `WavClip`, and planned ranges.

- [ ] Add fake-source tests asserting one bulk JSONB call per unique audio-ID set, segment-ID drift failure, deduplicated target/context/pair ranges, preserved request ordering, and queue limits by planned batches and estimated bytes.
- [ ] Add audio tests for WAV decode, mono downmix, resampling to configured rate, finite normalization, hop-300 mel geometry, and actionable errors containing `SegmentKey`.
- [ ] Implement batch resolution inside short `database_session()` scopes; never share sessions across workers. Verify current segment index and ID before building range requests.
- [ ] Implement a bounded background producer that marks a batch consumed only when the trainer accepts it; cancellation closes sessions and queue workers reliably.
- [ ] Re-run tests; expect PASS and bulk-call counters of one per refill.
- [ ] Commit: `git commit -m 'feat: bulk prefetch beetle audio'`.

### Task 5: Collation, augmentations, and pipeline composition

**Files:**
- Create: `src/runner/nodes/training/beetle/data/collate.py`
- Create: `src/runner/nodes/training/beetle/data/pipeline.py`
- Modify temporarily: `/tmp/test_beetle_data.py`

**Interfaces:**
- Produces: `collate_examples(examples, config) -> BeetleBatch`, `build_data_pipeline(config, stage, callbacks)`, `DataPipeline.next_batch()`, `DataPipeline.state_dict()`, and `DataPipeline.load_state_dict(state)`.
- Consumes: all earlier data interfaces and phoneme/ALBERT tokenizers supplied by runtime.

- [ ] Add failing mixed-batch tests with every combination of absent/present pre/post context and prompts. Assert independent padding/masks, correct duration/alignment shapes, stable order, and pinned tensors.
- [ ] Add deterministic augmentation tests for time stretch, pitch shift, and gain on embedding views while target waveform, mel, transcript, phonemes, and alignment remain byte-identical.
- [ ] Implement whole-batch feature preparation, tokenizer calls, per-field padding, and explicit bool masks. Do not encode missing context as present zero data.
- [ ] Compose index, planner, source, prefetch, and collator; state must include planner position and data fingerprint but exclude prefetched-unconsumed advancement.
- [ ] Run the complete temporary data suite; expect PASS, then remove `/tmp/test_beetle_data.py` with `apply_patch`.
- [ ] Run `wc -l` on every data file and `git diff --check`; expect all files below 300 lines.
- [ ] Commit: `git commit -m 'feat: complete beetle data pipeline'`.
