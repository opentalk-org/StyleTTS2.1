# Beetle Multi-GPU and Audio Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact-resumable Accelerate data parallelism and remove the measured 3x Stage 1 audio-fetch slowdown.

**Architecture:** Every rank computes the same deterministic global batch plan and consumes a disjoint per-rank slice containing complete voice/style groups. A persistent cache-backed bulk reader fetches each cold WAV with one concurrent range request, while one ordered preparation worker per rank keeps a bounded batch queue full.

**Tech Stack:** Python 3.12, PyTorch 2.11+, Hugging Face Accelerate 1.14+, boto3, PostgreSQL shared CRUD, Pydantic v2, Nix.

## Global Constraints

- `batch_size`, `voices_per_batch`, and `recordings_per_batch` are per GPU.
- Same-voice and same-recording groups never split across ranks.
- Training exposes no epoch configuration, metric, counter, or termination condition.
- Exact resume requires the same world size and preserves unconsumed prefetch semantics.
- Only the main rank writes MLflow runs, artifacts, and shared checkpoint manifests.
- Use shared audio CRUD/storage boundaries; do not issue runner-owned SQLAlchemy queries.
- Keep temporary tests under `/tmp` and remove them before completion.
- Keep project-owned Python files below 300 lines.

---

### Task 1: One-read cached bulk WAV retrieval

**Files:**
- Create: `src/shared/db/audio/ranges/cache.py`
- Create: `src/shared/db/audio/ranges/reader.py`
- Modify: `src/shared/db/audio/ranges/bulk.py`
- Modify: `src/shared/db/audio/ranges/__init__.py`
- Modify: `src/runner/nodes/speaker_clustering/source.py`
- Test temporarily: `/tmp/test_audio_range_reader.py`

**Interfaces:**
- Produces: `StoredWavLocation`, `AudioFileCache`, and `BulkWavReader.read(session, requests) -> list[WavClip]`.
- Consumes: one persistent `S3ObjectStore`, `$XDG_CACHE_HOME/runflow/audio`, cache budget bytes, and fetch-worker count.

- [ ] **Step 1: Write a failing cold-read test** using a fake packed object containing three WAV files; request duplicate clips and assert ordered results, exactly one `read_range` per distinct audio file, zero `download` calls, and no remote seek reads.
- [ ] **Step 2: Run** `nix develop --command python -m pytest -q /tmp/test_audio_range_reader.py`; expect failure because `BulkWavReader` does not exist.
- [ ] **Step 3: Implement immutable storage locations and concurrent full-WAV reads.** The core interface is:

```python
@dataclass(frozen=True)
class StoredWavLocation:
    audio_file_id: UUID
    object_path: str
    byte_offset: int
    byte_length: int

class BulkWavReader:
    def read(self, session: Session, requests: tuple[SegmentReadRequest, ...]) -> list[WavClip]: ...
```

Resolve locations with `get_audio_files_bulk`, deduplicate audio IDs, fetch full WAV slices concurrently, then call `slice_wav_ranges` for all requests belonging to each file.
- [ ] **Step 4: Add a failing cross-process cache test** where two spawned processes request one location simultaneously and assert one completed cache entry, no partial file, and one backing-store read.
- [ ] **Step 5: Implement the bounded cache** with a location hash, per-entry `fcntl` lock, temporary file plus atomic rename, a global eviction lock, and least-recently-used eviction that skips locked entries.
- [ ] **Step 6: Re-run the temporary suite** and expect all range, ordering, concurrency, corruption, and budget cases to pass.
- [ ] **Step 7: Update the speaker-clustering caller** to construct/reuse `BulkWavReader` without importing Beetle code.
- [ ] **Step 8: Commit** with `git commit -m 'perf: cache bulk wav range reads'`.

### Task 2: Rank-sharded voice-aware batch planning

**Files:**
- Modify: `src/runner/nodes/training/beetle/data/sampling.py`
- Modify: `src/runner/nodes/training/beetle/data/prefetch.py`
- Modify: `src/runner/nodes/training/beetle/data/pipeline.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`
- Test temporarily: `/tmp/test_beetle_distributed_sampling.py`

**Interfaces:**
- Produces: `DistributedShard(rank: int, world_size: int)` and a planner whose state is identical on every rank after each global draw.
- Consumes: per-rank batch/group sizes and the existing deterministic permutation pools.

- [ ] **Step 1: Write failing two-rank tests** over 5,000 synthetic segment keys: each rank receives 2,500 distinct targets, unions cover the dataset, local batch size is unchanged, planner states match, and complete voice/style groups remain local.
- [ ] **Step 2: Run** `nix develop --command python -m pytest -q /tmp/test_beetle_distributed_sampling.py`; expect constructor/signature failures.
- [ ] **Step 3: Implement rank slicing after global planning:**

```python
@dataclass(frozen=True)
class DistributedShard:
    rank: int
    world_size: int

global_count = self.batch_size * self.shard.world_size
local_start = self.shard.rank * self.batch_size
local_plans = tuple(global_plans[local_start:local_start + self.batch_size])
```

Draw global voice/style group counts, then slice whole groups with the same rule. Include `world_size` in `DataPipelineState` and reject incompatible resume.
- [ ] **Step 4: Verify deterministic padding/wrap behavior** for dataset sizes not divisible by world size and exact state restoration from every training phase.
- [ ] **Step 5: Re-run the temporary suite** and expect all sharding and resume assertions to pass.
- [ ] **Step 6: Commit** with `git commit -m 'feat: shard beetle batches across ranks'`.

### Task 3: Ordered audio-prefetch pipeline

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/data.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config.yaml`
- Modify: `src/runner/nodes/training/beetle/data/source.py`
- Modify: `src/runner/nodes/training/beetle/data/pipeline.py`
- Split if needed: `src/runner/nodes/training/beetle/data/prefetch.py`
- Test temporarily: `/tmp/test_beetle_audio_prefetch.py`

**Interfaces:**
- Produces: one ordered preparation worker per rank backed by `BulkWavReader` and a bounded prepared-batch queue.
- Consumes: required config fields `audio_cache_bytes` and `audio_fetch_workers`; removes `maximum_full_read_bytes` and complete-batch `worker_count`.

- [ ] **Step 1: Write failing pipeline tests** proving queue look-ahead, one collator active at a time, cancellation during fetch/queue waits, ordered output, byte-budget enforcement, and unconsumed-prefetch exclusion from checkpoint state.
- [ ] **Step 2: Run** `nix develop --command python -m pytest -q /tmp/test_beetle_audio_prefetch.py`; expect strict-config and loader failures.
- [ ] **Step 3: Add explicit configuration:**

```yaml
prefetch:
  page_size: 2000
  planned_batches: 8
  decoded_bytes: 2147483648
  audio_cache_bytes: 8589934592
  audio_fetch_workers: 16
```

- [ ] **Step 4: Rewire `DatabaseBatchSource`** to own one persistent reader and make the producer prepare batches sequentially while the reader performs concurrent cold I/O internally.
- [ ] **Step 5: Re-run the temporary suite** and expect all ordering, cancellation, bounds, and exact-state cases to pass.
- [ ] **Step 6: Commit** with `git commit -m 'perf: prefetch beetle training audio'`.

### Task 4: Accelerate data-parallel training runtime

**Files:**
- Modify: `pyproject.toml`
- Update: `uv.lock`
- Create: `src/runner/nodes/training/beetle/training/distributed.py`
- Modify: `src/runner/nodes/training/beetle/training/optimizer.py`
- Modify: `src/runner/nodes/training/beetle/training/checkpoint.py`
- Modify: `src/runner/nodes/training/beetle/training/loop_events.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/stages.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/services.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2.py`
- Modify: `src/runner/nodes/training/beetle/training/stage3.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1_setup.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_setup.py`
- Test temporarily: `/tmp/test_beetle_accelerate.py`

**Interfaces:**
- Produces: `DistributedRuntime` wrapping `Accelerator`, rank-aware callbacks/reporting/checkpoint coordination, prepared modules/optimizers, metric reduction, and rank RNG snapshots.
- Consumes: existing stage trainers and checkpoint payloads without changing model/loss behavior.

- [ ] **Step 1: Add direct `accelerate>=1.14` dependency** and update the lock through the existing Nix-wrapped uv workflow.
- [ ] **Step 2: Write a failing two-process CPU test** launched with `python -m accelerate.commands.launch --cpu --num_processes 2`: each rank observes a different sample shard, gradients/model parameters agree after one update, only rank zero writes the artifact, and a resume reproduces the next update.
- [ ] **Step 3: Run the launcher test through Nix** and expect missing distributed-runtime failures.
- [ ] **Step 4: Implement the runtime boundary:**

```python
class DistributedRuntime:
    @property
    def shard(self) -> DistributedShard: ...
    def prepare_module(self, module: nn.Module) -> nn.Module: ...
    def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer: ...
    def backward(self, loss: Tensor) -> None: ...
    def reduce_metrics(self, metrics: tuple[TrainingMetric, ...]) -> tuple[TrainingMetric, ...]: ...
    def gather_rank_state(self, state: RankState) -> tuple[RankState, ...] | None: ...
```

Use `Accelerator` for device selection, autocast, backward, optimizer wrapping, clipping, barriers, and failure propagation. Prepared submodules must still be called through their wrappers; use `unwrap_model` for state capture and configuration access.
- [ ] **Step 5: Make checkpoint version 5 rank-aware.** Store `world_size` and ordered rank RNG states, require the same world size on resume, gather state before main-rank atomic save, broadcast the completed checkpoint path, and keep non-main ranks from MLflow/artifact writes.
- [ ] **Step 6: Re-run the two-process suite** and expect synchronized parameters, disjoint samples, single-writer behavior, cancellation propagation, and exact resume.
- [ ] **Step 7: Commit** with `git commit -m 'feat: add accelerate beetle training'`.

### Task 5: Real graph verification and performance gate

**Files:**
- Modify: `src/runner/nodes/training/beetle/README.md`
- Modify: `src/runner/nodes/training/beetle/main.md`
- Verify: `src/shared/db/audio/ranges/`
- Verify: `src/runner/nodes/training/beetle/`
- Create temporarily: `/tmp/beetle_multigpu_benchmark/`

**Interfaces:**
- Produces: documented single-/multi-GPU commands and measured evidence that real prefetch no longer dominates Stage 1.
- Consumes: the approved LJSpeech config/checkpoint and real graph/runner path.

- [ ] **Step 1: Run all temporary suites together** through `nix develop --command python -m pytest -q ...`; expect PASS, then run `python -m compileall -q` and `git diff --check`.
- [ ] **Step 2: Run the cold-cache reader probe** on a real 64-file batch; expect at most 64 range requests, zero whole-pack downloads, preserved clips, and a warm-cache rerun with zero object reads.
- [ ] **Step 3: Run the Stage 1 performance benchmark** with batch size 64 per GPU, BF16, compilation, 9,600-sample crops, 10 warmups, and 30 measured steps. Expect real-prefetch mean within 20% of the cached-CPU-batch baseline.
- [ ] **Step 4: Run the distributed Stage 1 smoke.** When two CUDA devices are visible, launch one rank per GPU through Accelerate and the existing runner/API path. On a one-GPU host, run the two-rank CPU integration plus the single-GPU graph separately and record the hardware limitation. Inspect logs for disjoint sample keys, synchronized optimizer steps, rank system metrics, one MLflow run, validation artifacts, and one atomic checkpoint.
- [ ] **Step 5: Stop the smoke run cleanly, resume it, and verify** the next step, sampler position, loss schedule, optimizer/model state, and per-rank RNG continue without loss.
- [ ] **Step 6: Document launch/config/cache behavior**, remove all temporary tests and benchmark files, verify project file limits, and inspect `git status --short`.
- [ ] **Step 7: Commit** with `git commit -m 'docs: document beetle distributed training'`.
