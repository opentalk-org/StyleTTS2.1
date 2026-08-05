# FineWiki TTS Synthesis Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize all 101,250 FineWiki lines through fast backend nodes, atomically persist each clip into `tts_piper` or `tts_kokoro`, and finish with a measured projection below five hours.

**Architecture:** Two resumable input/synthesis nodes read deterministic corpus jobs and emit bounded audio batches. The existing audio-record writer is generalized to commit record creation and dataset membership together; stable metadata keys let reruns skip completed clips.

**Tech Stack:** Python 3.12, Runflow nodes, Piper/ONNX Runtime, Kokoro/PyTorch, SQLAlchemy/PostgreSQL, NATS runner, S3-compatible packed audio storage.

## Global Constraints

- Run every Python command through `nix develop --command python`.
- Test nodes only through a submitted graph; direct node `execute()` calls are forbidden.
- Preserve every unrelated audio record and dataset membership.
- Every output must have both dataset membership and matching `tts_dataset` metadata.
- The complete plan contains exactly 101,250 unique source keys.
- Keep files below 300 lines and folders below 16 files.
- Remove temporary tests before completion.

---

### Task 1: Atomic dataset-tagged audio persistence

**Files:**
- Modify: `src/runner/nodes/audio_segments/writeback.py`
- Modify: `src/shared/db/datasets/crud.py`
- Modify: `src/shared/db/datasets/__init__.py`
- Test temporarily: `/tmp/test_tts_dataset_persistence.py`

**Interfaces:**
- Produces: `SaveAudioRecordSettings.dataset_id: UUID | None`.
- Produces: `list_dataset_metadata_values(session, dataset_id, key) -> set[str]`.
- Guarantees: stored audio creation and dataset membership use one transaction.

- [ ] **Step 1: Write failing CRUD and settings tests**

Create a temporary test that verifies the settings accept `dataset_id`, the
dataset CRUD returns JSON metadata values only for members of the requested
dataset, and a forced membership failure rolls back the new audio row.

- [ ] **Step 2: Verify red**

Run:

```bash
nix develop --command python -m unittest /tmp/test_tts_dataset_persistence.py -v
```

Expected: failure because `dataset_id` and
`list_dataset_metadata_values` do not exist.

- [ ] **Step 3: Implement the generalized transaction**

Add:

```python
class SaveAudioRecordSettings(StrictSettings):
    storage_mode: Literal["stored", "external"] = "stored"
    virtual: bool = False
    bulk_import_packs: bool = False
    dataset_id: UUID | None = None
```

For stored mode, call `bulk_create_audio_files(..., commit=False)`, then
`bulk_add_audio_files_to_dataset(..., commit=False)`, then `session.commit()`.
Reject `dataset_id` with external mode because that path has a separate
transaction contract.

Add one generic JSON metadata-value query to the dataset CRUD facade.

- [ ] **Step 4: Verify green and graph registration**

Run the temporary test and export `/schema`; confirm the `SaveAudioRecord`
settings include nullable `dataset_id`.

- [ ] **Step 5: Commit**

```bash
git add src/runner/nodes/audio_segments/writeback.py src/shared/db/datasets
git commit -m "feat: atomically tag saved audio with a dataset"
```

### Task 2: Deterministic corpus planning and resume

**Files:**
- Create: `src/runner/nodes/tts/corpus/__init__.py`
- Create: `src/runner/nodes/tts/corpus/models.py`
- Create: `src/runner/nodes/tts/corpus/plan.py`
- Test temporarily: `/tmp/test_tts_corpus_plan.py`

**Interfaces:**
- Produces: immutable `CorpusJob`, `PiperModelPlan`, and `CorpusPlan`.
- Produces: `build_corpus_plan(root, piper_catalog) -> CorpusPlan`.
- Produces: `without_completed(jobs, completed_keys) -> tuple[CorpusJob, ...]`.

- [ ] **Step 1: Write failing plan tests**

Load the real manifest and assert:

```python
assert len(plan.piper_jobs) == 71_100
assert len(plan.kokoro_jobs) == 30_150
assert len(plan.source_keys) == 101_250
assert len(set(plan.source_keys)) == 101_250
```

Also assert each TXT line appears once, Japanese Piper streams route to Kokoro,
unsupported registered languages route to Piper, and completed keys filter out.

- [ ] **Step 2: Verify red**

Run:

```bash
nix develop --command env PYTHONPATH=src python -m unittest /tmp/test_tts_corpus_plan.py -v
```

Expected: import failure for `runner.nodes.tts.corpus.plan`.

- [ ] **Step 3: Implement immutable planning**

Read `manifest.json`, validate every declared path and line count, select the
largest-speaker Piper model per language, rotate speaker IDs and Kokoro presets
by stream index, then build stable keys:

```python
source_key = f"{engine}:{stream_id}:{sentence_index:04d}"
```

Reject missing, extra, empty, or duplicate lines and any plan total other than
101,250.

- [ ] **Step 4: Verify green**

Run the plan tests twice and compare serialized plans to prove determinism.

- [ ] **Step 5: Commit**

```bash
git add src/runner/nodes/tts/corpus
git commit -m "feat: plan resumable FineWiki TTS corpus jobs"
```

### Task 3: High-throughput Piper and Kokoro corpus nodes

**Files:**
- Create: `src/runner/nodes/tts/corpus/audio.py`
- Create: `src/runner/nodes/tts/corpus/piper.py`
- Create: `src/runner/nodes/tts/corpus/kokoro.py`
- Modify: `src/runner/nodes/tts/engines/piper.py`
- Modify: `src/runner/nodes/tts/__init__.py`
- Test temporarily: `/tmp/test_tts_corpus_nodes.py`

**Interfaces:**
- Produces: registered nodes `PiperCorpusSynthesis` and
  `KokoroCorpusSynthesis`.
- Both settings consume `corpus_dir`, `dataset_id`, `dataset_name`, and batch
  controls.
- Both output `audio: AudioPort(mode=STREAM)`.

- [ ] **Step 1: Write failing schema and audio-metadata tests**

Assert both nodes register, use no inputs, declare precise resources, emit
streaming audio, generate stable names, and attach `tts_source_key`,
`tts_dataset`, engine, voice, language, text, stream, and sentence index.

- [ ] **Step 2: Verify red**

Run:

```bash
nix develop --command env PYTHONPATH=src python -m unittest /tmp/test_tts_corpus_nodes.py -v
```

Expected: missing node imports.

- [ ] **Step 3: Implement Piper parallelism**

Create 15 LPT-balanced worker shards over whole voice files. Configure each
Piper ONNX session with one intra-op and one inter-op thread. Keep one runtime
per worker/model transition, synthesize bounded contiguous groups concurrently,
check cancellation between groups, and report completed/total progress.

- [ ] **Step 4: Implement Kokoro lifecycle batching**

Resolve/download the Kokoro catalog checkpoint during setup, load one runtime,
synthesize bounded jobs, check cancellation per item, report progress, and
release accelerator memory during teardown.

- [ ] **Step 5: Implement resume filtering**

During setup, read existing `tts_source_key` values from the configured dataset
and remove them from each node's pending plan.

- [ ] **Step 6: Verify green and schema**

Run unit tests and inspect `/schema` after restarting the shared dev session.

- [ ] **Step 7: Commit**

```bash
git add src/runner/nodes/tts
git commit -m "feat: add high-throughput TTS corpus synthesis nodes"
```

### Task 4: Real graph benchmark, campaign launch, and audit

**Files:**
- Create: `imports/run_tts_corpus_campaign.py`
- Create: `workflows/tts_finewiki_corpus.json`
- Test temporarily: `/tmp/audit_tts_corpus.py`

**Interfaces:**
- Produces datasets `tts_piper` and `tts_kokoro`.
- Produces one resumable graph with both corpus synthesis branches feeding
  dataset-tagged `SaveAudioRecord` nodes.

- [ ] **Step 1: Implement idempotent launcher**

The launcher gets or creates both datasets through the backend, builds a graph
with runtime resources `cpu_workers=15`, `accelerator=1`, and detected VRAM,
and supports `--benchmark-lines 90` or full mode. It never calls a delete API.

- [ ] **Step 2: Restart shared services and submit benchmark**

Run:

```bash
nix develop --command runflow-dev-session
nix develop --command python imports/run_tts_corpus_campaign.py --benchmark-lines 90
```

Inspect via:

```bash
nix develop --command python -m cli runs
nix develop --command python -m cli logs <run_id>
nix develop --command python -m cli failed <run_id>
```

Expected: successful graph, 180 tagged files, positive audio bytes/durations,
and measured projection below five hours.

- [ ] **Step 3: Tune only if projection misses**

Adjust Piper worker/group size or Kokoro batch size, rerun the same benchmark,
and retain only a configuration whose measured projection is below five hours.
Do not reduce corpus scope.

- [ ] **Step 4: Submit the complete campaign**

Run:

```bash
nix develop --command python imports/run_tts_corpus_campaign.py
```

Record the run ID and monitor it until terminal.

- [ ] **Step 5: Audit completion from PostgreSQL**

The audit compares planned source keys to dataset members and asserts exact
equality, 101,250 total memberships, matching `tts_dataset` flags, positive
durations/bytes, and uniqueness. It also compares pre/post unrelated audio
counts and memberships to prove no deletion.

- [ ] **Step 6: Remove temporary tests and commit durable launch artifacts**

```bash
gio trash /tmp/test_tts_dataset_persistence.py /tmp/test_tts_corpus_plan.py /tmp/test_tts_corpus_nodes.py /tmp/audit_tts_corpus.py
git add imports/run_tts_corpus_campaign.py workflows/tts_finewiki_corpus.json
git commit -m "feat: launch resumable FineWiki TTS corpus campaign"
```
