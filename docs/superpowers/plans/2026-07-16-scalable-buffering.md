# Scalable Buffering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep queued tasks visible until execution and make segment sourcing scale across many small and a few large stored WAV files.

**Architecture:** `runflow` observes its existing typed queues without staging tasks privately, using only generic count, byte-cost, deadline, and upstream-state signals. Audio range selection and WAV slicing live behind shared audio CRUD and are consumed by the runner source.

**Tech Stack:** Python 3.12, asyncio, Pydantic, SQLAlchemy, S3-compatible range reads, standard-library `wave`, Nix development shell.

## Global Constraints

- Keep `src/runflow` domain-agnostic.
- Do not add node-name or datatype special cases to the scheduler.
- Run every Python command through `nix develop --command`.
- Keep temporary regressions uncommitted and remove them before completion.
- Keep every file below 300 lines and every folder below 16 files.

---

### Task 1: Observable generic batch readiness

**Files:**
- Modify: `src/runflow/runtime/batch_collector.py`
- Modify: `src/runflow/runtime/scheduling/admission.py`
- Modify: `src/runflow/runtime/scheduler.py`
- Test: `_tmp_batch_buffer_regression.py` (remove after verification)

**Interfaces:**
- Consumes: `asyncio.Queue[Task]`, `BatchPolicy`, task byte weights, active ancestor counts, blocked admissions.
- Produces: `BatchCollector.collect(node: Node, first: Task) -> tuple[list[list[Task]], list[Task]]` with no hidden waiting tasks.

- [ ] Write a temporary scheduler regression whose slow producer emits five tasks, whose consumer prefers four, and which records queue-depth events before the consumer starts.
- [ ] Run it with `nix develop --command python _tmp_batch_buffer_regression.py`; expect failure because queue depth falls to zero while the collector is still waiting.
- [ ] Change collection to leave the first and subsequent tasks in the queue until a generic readiness condition fires, then dequeue and plan the batch atomically on the event loop.
- [ ] Add count-capacity blocking to the same generic admission-pressure signal already used for byte pressure.
- [ ] Re-run the regression; expect batches `[4, 1]`, visible waiting depths, and termination under a one-item byte budget.

### Task 2: Explicit completed work items

**Files:**
- Modify: `src/runflow/runtime/scheduler_events.py`
- Modify: `src/shared/event_store.py`
- Test: `_tmp_completed_items_regression.py` (remove after verification)

**Interfaces:**
- Produces: `batch_completed.detail["completed_items"]`, defined as emitted items for inputs and consumed items for transforms.

- [ ] Write a temporary event-store regression where a source task emits 128 items and assert `tasks_completed == 128`.
- [ ] Run it through Nix; expect the current value `1`.
- [ ] Emit and consume the explicit `completed_items` field without branching on concrete node types.
- [ ] Re-run; expect source `128` and transform completion equal to consumed batch size.

### Task 3: Adaptive stored-WAV segment reads

**Files:**
- Create: `src/shared/db/audio/ranges/__init__.py`
- Create: `src/shared/db/audio/ranges/wav.py`
- Modify: `src/shared/db/audio/crud.py`
- Modify: `src/runner/nodes/speaker_clustering/source.py`
- Test: `_tmp_wav_range_regression.py` (remove after verification)

**Interfaces:**
- Produces: typed segment-read requests and `bulk_read_wav_segments(...) -> list[bytes]` preserving request order.
- Consumes: audio-file rows and `ObjectStore.read_range(path, offset, length)`.

- [ ] Write temporary regressions using generated small and sparse large PCM WAV fixtures: small inputs use grouped full reads; a large input reads only header and requested frame ranges; output WAV frames match the requested intervals.
- [ ] Run through Nix; expect failure because the range API does not exist.
- [ ] Implement a seekable bounded object-range reader and parse each source WAV once per request group.
- [ ] Select full versus ranged reads from generic byte cost, never from file names or node types.
- [ ] Re-run; expect exact clip frames and transferred bytes bounded by requested ranges for the large fixture.

### Task 4: Output-cost source paging

**Files:**
- Modify: `src/runner/nodes/speaker_clustering/source.py`
- Test: `_tmp_speaker_source_regression.py` (remove after verification)

**Interfaces:**
- Consumes: `SegmentReference` duration and stored-audio metadata plus `bulk_read_wav_segments`.
- Produces: source pages bounded by `QUEUE_MAX_SIZE` and estimated resident clip bytes.

- [ ] Write a temporary graph regression with many small source files and several oversized source files, asserting pages are not reduced to one merely because source recordings are large.
- [ ] Run through Nix; expect the oversized case to emit singleton pages.
- [ ] Replace whole-source byte-prefix selection with output-clip cost selection and adaptive shared reads.
- [ ] Run the graph regression; expect bounded multi-item pages and exact output count.

### Task 5: Integrated verification

**Files:**
- Remove: all `_tmp_*_regression.py` files created by this plan.

- [ ] Run compile checks and all temporary regressions through Nix.
- [ ] Restart only the shared Zellij stack with `runflow-dev-stop` followed by `runflow-dev-session`.
- [ ] Submit the real speaker clustering workflow through `POST /graphs/runs` and inspect it with the CLI.
- [ ] Confirm queued tasks remain visible, source `done` counts emitted segments, source pages do not collapse to one because of large recordings, and embed batches reach 128 when sufficient work remains.
- [ ] Remove temporary regressions, run `git diff --check`, confirm scoped files remain under size limits, and leave the shared stack running.
