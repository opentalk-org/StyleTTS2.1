# StyleTTS Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the legacy StyleTTS Studio service behavior into v2 runner graph nodes in the order that unlocks real workflows with shared DB and bucket-backed data.

**Architecture:** Keep `src/runflow` domain-agnostic. Put StyleTTS and audio behavior under `src/runner/nodes`, and keep persistence behind `src/shared/db/*/crud.py` facades. Split legacy monolithic jobs into transform nodes plus explicit writeback nodes, except for long GPU training loops that remain coarse execution nodes with separate asset/config preparation.

**Tech Stack:** Python 3.12, Pydantic, FastAPI, SQLAlchemy, PostgreSQL, shared S3/RustFS object storage, React/Vite/Tailwind for workflow templates and schema-driven UI.

---

## Scope And Constraints

- Do not add tests. `AGENTS.md` says: "THERE IS NO TESTS AND DONT ADD THEM."
- Verify with `python -m compileall`, `/schema` export smoke checks, frontend `npm run build`, and local graph smoke runs.
- Keep each file under 300 lines and each folder under 16 files.
- Do not add audio-specific behavior to `src/runflow`.
- Do not bypass shared CRUD facades for PostgreSQL or bucket-backed audio/assets.
- Current dirty files were present before this plan; record `git status --short` before execution and do not overwrite unrelated changes.

## File Structure Plan

- `src/runner/nodes/models.py` - shared dataclasses for node payloads.
- `src/runner/nodes/datatypes.py` - runflow datatype registration.
- `src/runner/nodes/registry.py` - import and register node classes.
- `src/runner/nodes/audio_io.py` - source/load/save audio and transcript IO nodes; split if it approaches 260 lines.
- `src/runner/nodes/audio_processing.py` - keep only lightweight coordination or move existing classes into focused audio submodules if implementation grows.
- `src/runner/nodes/audio_segments/` - new package for segment payloads, segment load/writeback, grouping, and split extraction nodes.
- `src/runner/nodes/audio_enhancement/` - new package for normalize and denoise implementation helpers.
- `src/runner/nodes/statistics/` - new package for per-file features, aggregation, and statistics save nodes.
- `src/runner/nodes/assets/` - new package for checkpoint, catalog, and extra-file resolution nodes.
- `src/runner/nodes/training/` - new package for manifest/config/publish helpers if `training.py` crosses 300 lines.
- `src/runner/nodes/synthesis/` - new package for style reference and StyleTTS synthesis nodes if `testing.py` crosses 300 lines.
- `src/shared/db/audio/segments_crud.py` - segment-specific CRUD helpers so `audio/crud.py` stays under 300 lines.
- `src/shared/db/statistics/` - add only if no shared statistics facade exists when implementing statistics persistence.
- `src/frontend/src/features/workflows/templates.ts` - add production workflow templates after backend schema exposes the needed nodes.

## Legacy Anchors

- Upload/import side effects: `tmp/styletts_studio/src/backend/app/services/upload/jobs.py`
- Normalize: `tmp/styletts_studio/src/backend/app/services/normalize_audio/process.py`
- Denoise: `tmp/styletts_studio/src/backend/app/services/denoise/deepfilter2.py`
- ASR modes: `tmp/styletts_studio/src/backend/app/services/transcribe/job_handlers.py`
- Phonemize: `tmp/styletts_studio/src/backend/app/services/phonemize/jobs.py`
- Split grouping/persist: `tmp/styletts_studio/src/backend/app/services/split_audio/groups.py` and `job_write.py`
- Statistics: `tmp/styletts_studio/src/backend/app/services/statistics/jobs.py` and `result_payload.py`
- Catalog/assets: `tmp/styletts_studio/src/backend/app/services/styletts2_catalog/`
- Training manifest/config/publish: `tmp/styletts_studio/src/backend/app/services/styletts_finetune/`
- Synthesis: `tmp/styletts_studio/src/backend/app/services/styletts_finetune_test_synth/`

---

### Task 1: Baseline, Package Boundaries, And Node Payload Types

**Files:**
- Modify: `src/runner/nodes/models.py`
- Modify: `src/runner/nodes/datatypes.py`
- Modify: `src/runner/nodes/registry.py`
- Create: `src/runner/nodes/audio_segments/__init__.py`
- Create: `src/runner/nodes/audio_enhancement/__init__.py`
- Create: `src/runner/nodes/statistics/__init__.py`
- Create: `src/runner/nodes/assets/__init__.py`

- [ ] Run `git status --short` and record unrelated dirty files in the task notes.
- [ ] Add dataclasses for `AudioSegment`, `SegmentGroup`, `CheckpointRef`, `AssetBundleRef`, `TrainingManifest`, `TrainingResult`, `SynthesisRequest`, and `SynthesisResult`.
- [ ] Register new datatypes: `AUDIO_SEGMENT`, `SEGMENT_GROUP`, `CHECKPOINT_REF`, `ASSET_BUNDLE`, `TRAINING_MANIFEST`, `TRAINING_RESULT`, `SYNTHESIS_RESULT`.
- [ ] Keep the existing `AUDIO_REF`, `AUDIO`, `TRANSCRIPT`, and `SAVE_RESULT` names stable.
- [ ] Run `python -m compileall src/runner src/runflow`.
- [ ] Smoke schema export with:

```bash
python - <<'PY'
from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.registry import register_runner_nodes, register_runner_types_for_ui
nodes = register_runner_nodes(NodeRegistry()).to_schema()
types = register_runner_types_for_ui(TypeRegistry()).to_schema()
assert "AUDIO_REF" in types
assert "LoadAudio" in nodes
print(len(nodes), len(types))
PY
```

Expected: command exits 0 and prints node/type counts.

### Task 2: Shared Segment CRUD And Audio Writeback Nodes

**Files:**
- Create: `src/shared/db/audio/segments_crud.py`
- Modify: `src/shared/db/audio/crud.py`
- Create: `src/runner/nodes/audio_segments/writeback.py`
- Modify: `src/runner/nodes/audio_io.py`
- Modify: `src/runner/nodes/registry.py`

- [ ] Move or wrap segment CRUD operations from `audio/crud.py` into `segments_crud.py`: create segment, replace all segments, update segment text, update segment phonemes, delete segment.
- [ ] Keep `audio/crud.py` public calls working by importing wrappers from `segments_crud.py` if current backend routes still use them.
- [ ] Add `SaveAudioRecordNode` for creating a new packed audio file through `shared.db.audio.crud.create_audio_file`.
- [ ] Add `UpdateAudioRecordBytesNode` for in-place audio byte replacement through `shared.db.audio.crud.update_audio_file`.
- [ ] Add `LoadAudioSegmentsNode`, `SaveAudioSegmentsNode`, `UpdateSegmentTextNode`, and `UpdateSegmentPhonemesNode`.
- [ ] Register all writeback nodes under categories `Audio / Segments` and `Audio / Writeback`.
- [ ] Run `python -m compileall src/shared src/runner src/backend`.
- [ ] Run the schema export smoke from Task 1 and assert the new node types appear.

### Task 3: Transcript And ASR Writeback Flow

**Files:**
- Modify: `src/runner/nodes/asr.py`
- Modify: `src/runner/nodes/text_processing.py`
- Create: `src/runner/nodes/audio_segments/transcripts.py`
- Modify: `src/runner/nodes/registry.py`

- [ ] Extend ASR settings with `scope: full|segments`, `segment_mode: replace|add`, and `segment_batch_size`.
- [ ] Implement `TranscriptToSegmentsNode` to convert full-file timestamped transcript spans into `AudioSegment` values.
- [ ] Implement `ApplyTranscriptToSegmentsNode` for existing-segment replace/add behavior without directly writing to DB.
- [ ] Implement real `PhonemizeTranscriptNode` output fields for phonemes instead of metadata-only annotation.
- [ ] Add `PhonemizeSegmentsNode` for segment streams with `fill|replace`, language, tie, punctuation, worker, and thread settings.
- [ ] Keep model internals stubbed only where dependencies are unavailable, but preserve output shape and lineage.
- [ ] Run `python -m compileall src/runner src/runflow`.
- [ ] Smoke a small inline graph with `SelectedAudioSource -> LoadAudio -> WhisperTranscribe -> TranscriptToSegments`; expected failure is only missing DB data if no audio IDs are supplied.

### Task 4: Normalize And Denoise As Real Audio Transforms

**Files:**
- Create: `src/runner/nodes/audio_enhancement/normalize.py`
- Create: `src/runner/nodes/audio_enhancement/denoise.py`
- Modify: `src/runner/nodes/audio_processing.py`
- Modify: `src/runner/nodes/registry.py`

- [ ] Port the normalize algorithm from legacy `normalize_audio/process.py` into a helper that accepts bytes and returns normalized WAV bytes plus measured duration and leading padding seconds.
- [ ] Make `NormalizeLoudnessNode` return a new `Audio` value with updated bytes, duration metadata, and `leading_pad_seconds`.
- [ ] Port DeepFilterNet load/enhance/save logic behind `DeepFilterNetDenoiseNode.setup()` and `execute()`.
- [ ] Keep denoise resource policy as `{"accelerator": 1, "vram_gb": 4}` and load the model through node lifecycle, not inside each item loop.
- [ ] Add a workflow path for `LoadAudio -> DeepFilterNetDenoise -> NormalizeLoudness -> UpdateAudioRecordBytes`.
- [ ] Run `python -m compileall src/runner src/shared src/runflow`.

### Task 5: Split Audio Workflow Nodes

**Files:**
- Create: `src/runner/nodes/audio_segments/grouping.py`
- Create: `src/runner/nodes/audio_segments/extract.py`
- Modify: `src/runner/nodes/registry.py`
- Modify: `src/runner/nodes/datatypes.py`

- [ ] Port `iter_split_groups`, dynamic IPA minimum sampling, and text/phoneme merge helpers into `grouping.py`.
- [ ] Add `PlanSegmentGroupsNode` with settings: mode, min total seconds, min IPA chars, max gap, max IPA chars, max merged duration.
- [ ] Add `ExtractSegmentGroupAudioNode` using ffmpeg or a structured audio library helper to produce `Audio` values for grouped spans.
- [ ] Add `PersistSplitAudioRecordsNode` to create new audio records, create adjusted child segments, attach target/source datasets, and optionally delete source records for `replace_all`.
- [ ] Ensure lineage links each group back to the source audio ID.
- [ ] Run `python -m compileall src/runner src/shared src/runflow`.

### Task 6: Dataset Statistics Nodes

**Files:**
- Create: `src/shared/db/statistics/__init__.py`
- Create: `src/shared/db/statistics/models.py`
- Create: `src/shared/db/statistics/schemas.py`
- Create: `src/shared/db/statistics/crud.py`
- Create: `src/runner/nodes/statistics/audio_features.py`
- Create: `src/runner/nodes/statistics/aggregate.py`
- Create: `src/runner/nodes/statistics/writeback.py`
- Modify: `src/runner/nodes/audio_processing.py`
- Modify: `src/runner/nodes/registry.py`

- [ ] Port per-file feature extraction from legacy `statistics/audio_features.py`.
- [ ] Add a shared statistics CRUD facade with `StatisticsEntry` create/list/read functions and a Pydantic payload schema for persisted result JSON.
- [ ] Add `AnalyzeAudioFeaturesNode` for `Audio -> JSON` feature records.
- [ ] Add `AggregateDatasetStatisticsNode` with list input support for feature records and segment text/phoneme metadata.
- [ ] Add `SaveStatisticsEntryNode` that persists aggregate payloads only through `src/shared/db/statistics/crud.py`.
- [ ] Keep `CalculateAudioStatsNode` as a compatibility alias only if workflow templates still reference it.
- [ ] Run `python -m compileall src/runner src/shared src/backend`.
- [ ] Export schema and assert `AnalyzeAudioFeatures`, `AggregateDatasetStatistics`, and `SaveStatisticsEntry` are visible.

### Task 7: Catalog, Checkpoint, And Asset Resolution Nodes

**Files:**
- Create: `src/runner/nodes/assets/catalog.py`
- Create: `src/runner/nodes/assets/checkpoints.py`
- Create: `src/runner/nodes/assets/training_assets.py`
- Modify: `src/runner/nodes/training.py`
- Modify: `src/runner/nodes/testing.py`
- Modify: `src/runner/nodes/registry.py`

- [ ] Add `CatalogDownloadNode` for `styletts2_utils`, `official_checkpoints`, `papercup_multilingual_pl_bert`, and `vokan_checkpoint`.
- [ ] Add `ResolveCheckpointNode` to validate checkpoint ID and type and return a `CheckpointRef` with cached local path from `shared.db.assets.crud.get_checkpoint_path`.
- [ ] Add `ResolveTrainingAssetsNode` to resolve ASR bundle, F0 model, PL-BERT, and OOD text sets.
- [ ] Update existing `SelectCheckpoint`, `PrefetchCheckpoint`, `SelectTrainingAssets`, and `PrefetchTrainingAssets` stubs to use the new typed asset refs.
- [ ] Run `python -m compileall src/runner src/shared src/backend`.

### Task 8: Training Preparation And Training Nodes

**Files:**
- Create: `src/runner/nodes/training_manifest.py`
- Create: `src/runner/nodes/training_config.py`
- Modify: `src/runner/nodes/training.py`
- Modify: `src/runner/nodes/registry.py`

- [ ] Add `BuildTrainingManifestNode` to produce train/validation manifest files from dataset audio with phonemized segments.
- [ ] Add `BuildStyleTtsFinetuneConfigNode` to combine manifest, base checkpoint, ASR/F0/PL-BERT assets, OOD text sets, decoder, precision, and epoch settings.
- [ ] Replace `StyleTtsFinetuneNode` stub output with execution that consumes the prepared config and publishes checkpoints through shared asset CRUD.
- [ ] Replace `F0ModelTrainingNode` and `AsrModelTrainingNode` stubs with real execution paths that consume `TrainingManifest` and publish typed `TrainingResult`.
- [ ] Keep each long training node coarse-grained; do not create graph nodes for every internal epoch or optimizer step.
- [ ] Run `python -m compileall src/runner src/shared src/backend`.

### Task 9: StyleTTS Synthesis Nodes

**Files:**
- Create: `src/runner/nodes/synthesis/style_reference.py`
- Create: `src/runner/nodes/synthesis/styletts.py`
- Modify: `src/runner/nodes/testing.py`
- Modify: `src/runner/nodes/registry.py`

- [ ] Add `ResolveStyleReferenceNode` for audio-file reference and base64 WAV reference, enforcing exactly one source.
- [ ] Update `TestingPromptPhonemizerNode` to produce typed phoneme payloads compatible with checkpoint symbols.
- [ ] Replace `StyleTtsSynthesisNode` stub with runtime load, checkpoint weight resolution, style encoding, synthesis, and `SynthesisResult` output.
- [ ] Replace `StyleTtsSweepSynthesisNode` stub with voice-to-reference selection and repeated synthesis output.
- [ ] Connect synthesis outputs to `SaveGeneratedAudio` or `SaveAudioRecord`.
- [ ] Run `python -m compileall src/runner src/shared src/backend`.

### Task 10: Workflow Templates And Smoke Verification

**Files:**
- Modify: `src/frontend/src/features/workflows/templates.ts`
- Modify: `src/frontend/src/features/testing/workflows.ts`
- Modify: `src/frontend/src/features/training/logic.ts`
- Modify: `src/frontend/src/features/testing/logic.ts`

- [ ] Update workflow templates to use explicit writeback nodes instead of local artifact-only saves for production flows.
- [ ] Add templates for: transcription to segments, normalize and update records, denoise-normalize-update, phonemize segments, split audio, dataset statistics, StyleTTS finetune, single synthesis, and sweep synthesis.
- [ ] Ensure frontend launch source compilation still patches only source nodes via `src/backend/workflows/service.py`.
- [ ] Run `npm run build` from `src/frontend`.
- [ ] Run `python -m compileall src/backend src/shared src/runner src/runflow`.
- [ ] Start the stack with `nix develop --command runflow-dev` and manually smoke `/schema`, one source/load graph, and one writeback-free graph. Stop with Ctrl-C.

## Completion Criteria

- `src/runflow` remains free of audio or StyleTTS vocabulary.
- All production nodes use shared CRUD for DB and bucket-backed audio/assets.
- Existing source/load nodes still work with selected audio, dataset audio, and all audio launch sources.
- Segment, transcript, audio update, statistics, training, and synthesis workflows are expressible as graph JSON without legacy job handlers.
- `python -m compileall src/backend src/shared src/runner src/runflow` passes.
- `npm run build` passes from `src/frontend`.
- `/schema` includes all new nodes and datatypes.
