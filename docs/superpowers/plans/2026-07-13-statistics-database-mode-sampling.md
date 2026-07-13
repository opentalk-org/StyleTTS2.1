# Statistics Database Mode and Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add database-only statistics and reusable `ALL`/random-`N` audio-reference sampling while preserving the existing aggregation and saved-statistics pathway.

**Architecture:** A `RandomAudioSubset` transform samples reference objects before any audio bytes are loaded and repairs source-batch metadata. A separate `DatabaseStatisticsFeatures` node emits the same feature-record contract as acoustic analysis without reading storage. The frontend builds one of two graph shapes and hides acoustic-only results when unavailable.

**Tech Stack:** Python 3.12, Pydantic, runflow typed ports and batching, React, TypeScript, TanStack Query, Tailwind CSS.

## Global Constraints

- Do not change `AudioSource`.
- Database-only mode must not read audio bytes, packs, S3 objects, or waveforms.
- `RandomAudioSubset` buffers at most `N` references in random mode.
- Sampling applies to every statistic in the saved result.
- Both modes feed `AggregateDatasetStatistics` and `SaveStatisticsEntry` through `feature_records`.
- Do not add committed test files, matching the user's standing instruction.

---

### Task 1: Random audio subset node

**Files:**
- Create: `src/runner/nodes/audio_sources/random_subset.py`
- Modify: `src/runner/nodes/audio_sources/__init__.py`
- Modify: `src/runner/nodes/registry.py`

**Interfaces:**
- Consumes: `audio: AudioPort()` carrying `source_batch_id` and `source_batch_count` metadata.
- Produces: `audio: AudioPort(mode=PortMode.STREAM)` with corrected subset batch metadata.

- [ ] Define settings with `selection: Literal["all", "random"]` and positive `count`.
- [ ] Pass through `all` mode without buffering.
- [ ] Implement per-source-batch reservoir sampling with disabled batching and cancellation checks.
- [ ] Emit the completed reservoir with a new source batch ID and emitted subset count.
- [ ] Register and schema-export the node.

### Task 2: Database feature-record node and shared payload flag

**Files:**
- Create: `src/runner/nodes/statistics/database_features.py`
- Modify: `src/runner/nodes/statistics/aggregate.py`
- Modify: `src/runner/nodes/registry.py`

**Interfaces:**
- Consumes: `audio: AudioPort()` containing segments loaded by `LoadAudioSegments`.
- Produces: `feature_records: JsonPort()` matching `AnalyzeAudioFeatures` output fields.

- [ ] Build database feature records from `Audio` identity, duration, metadata, and `speech_segment_records` only.
- [ ] Populate acoustic arrays with empty values and set `acoustic_metrics_available` false.
- [ ] Mark acoustic records from `AnalyzeAudioFeatures` as available.
- [ ] Add payload-level availability, computation mode, and sample scope fields in aggregation.
- [ ] Keep database-derived duration, corpus, segment, speaker, and voice aggregation common.

### Task 3: Statistics graph builder and compute controls

**Files:**
- Modify: `src/frontend/src/features/statistics/workflow.ts`
- Modify: `src/frontend/src/features/statistics/query.ts`
- Modify: `src/frontend/src/features/statistics/ComputeStatistics.tsx`

**Interfaces:**
- Produces: `computeDatasetStatistics(schema, datasetId, name, mode, sampleCount)`.

- [ ] Insert `RandomAudioSubset` after `AudioSource` in both graph modes.
- [ ] Build the database path without `LoadAudio` and the acoustic path with it.
- [ ] Add mode and `ALL`/random-count controls and validate a positive count.
- [ ] Pass the selected settings through the mutation into graph construction.

### Task 4: Viewer availability handling and graph verification

**Files:**
- Modify: `src/frontend/src/features/statistics/api.ts`
- Modify: `src/frontend/src/features/statistics/logic.ts`
- Modify: statistics rendering component files that consume `audioHistograms` and `audioTiles`.

**Interfaces:**
- Consumes: `acoustic_metrics_available`, `computation_mode`, and `sample_scope` from saved payloads.

- [ ] Extend the payload type with computation and sampling metadata.
- [ ] Hide clipping and waveform-derived charts for database-only entries while preserving duration and corpus content.
- [ ] Run Python compile checks and frontend TypeScript build through `nix develop --command`.
- [ ] Restart the shared stack and run database-only `ALL` and random-`N` graphs through `POST /graphs/runs`.
- [ ] Inspect node logs to confirm the database-only path contains no `LoadAudio` node or audio-byte reads.
