# High-Volume Persistence Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make high-volume audio, segment, waveform, statistics, voice, artifact, and dataset-membership node operations perform set-based persistence and one transaction per bounded node batch.

**Architecture:** Bulk CRUD functions are the source of truth for collection operations: they validate IDs set-wise, reuse one object-store client, group pack work, mutate in one transaction, and preserve ordered results. Runner nodes collect inputs, call one bulk facade, and map results back without changing lineage or fan-out semantics.

**Tech Stack:** Python 3.12, SQLAlchemy 2, PostgreSQL, Pydantic 2, S3-compatible storage, FastAPI runner graphs, Nix development shell.

## Global Constraints

- Keep `src/runflow` domain-agnostic; this plan does not modify it.
- Batch only high-cardinality workflow paths; keep dataset entities, settings, workflows, checkpoints, and other low-volume administrative CRUD singular.
- Treat dataset membership involving many audio IDs as high-volume audio work.
- Use one commit and one audio-pack prune pass per bounded bulk mutation.
- Validate missing IDs before mutation and include them in actionable exceptions.
- Preserve input order, output cardinality, lineage, cancellation, and fan-out behavior.
- Keep every file below 300 lines and every folder below 16 files.
- Use temporary tests/scripts only and remove them before completion.
- Run Python, CLI, services, and graph verification through `nix develop --command ...`.
- Work in the current checkout; do not create a branch or worktree.
- Execute inline because delegation was not requested.

## File Map

- Modify `src/shared/db/audio/pack_crud.py`: set-based packed row loading, grouped reads, updates, and deletes.
- Modify `src/shared/db/audio/crud.py`: collection get/update/delete orchestration and one prune pass.
- Modify `src/shared/db/waveforms/crud.py`: set-based waveform deletion and replacement cleanup.
- Modify `src/shared/db/datasets/crud.py`: bulk membership removal.
- Modify `src/shared/db/statistics/crud.py`: ordered bulk statistics creation.
- Modify `src/shared/db/voices/crud.py`: set-based voice lookup and creation.
- Modify `src/shared/db/assets/crud.py`: ordered high-volume extra-file creation with one store client and transaction.
- Modify `src/runner/nodes/dataset_writeback/nodes.py`: bulk membership, voice assignment, and deletion.
- Modify `src/runner/nodes/audio_io/nodes.py`: bulk audio loads and artifact creation.
- Modify `src/runner/nodes/audio_segments/writeback.py`: bulk byte updates.
- Modify `src/runner/nodes/audio_segments/extract.py`: bulk source loads and split persistence.
- Modify `src/runner/nodes/statistics/writeback.py`: bulk statistics persistence.
- Modify `src/runner/nodes/audio_segments/speaker_split.py`: bulk voice lookup/creation around already-batched diarization.
- Temporary `/tmp/test_persistence_batching.py`: transaction/query/order verification.
- Temporary `/tmp/delete_audio_records_graph.json`: real graph request.

---

### Task 1: Set-based packed audio reads and deletes

**Files:**
- Modify: `src/shared/db/audio/pack_crud.py`
- Modify: `src/shared/db/audio/crud.py`
- Test: `/tmp/test_persistence_batching.py`

**Interfaces:**
- Produce `get_audio_files_bulk(session: Session, audio_file_ids: Sequence[UUID]) -> dict[UUID, AudioFile]` in `audio/crud.py`.
- Keep `bulk_read_audio_files(...) -> dict[UUID, bytes]` and `bulk_delete_audio_files(...) -> None` public.
- Add private `_packed_items(session, ids) -> dict[UUID, AudioFile]` in `pack_crud.py`; it performs one `select(AudioFile).where(AudioFile.id.in_(ids))`, validates missing IDs, and returns an ID map.

- [ ] **Step 1: Write failing temporary tests** using a temporary PostgreSQL schema/session and a recording object store. Cover duplicate input IDs, missing-ID failure before mutation, two records sharing a pack causing one download, and multi-delete causing one commit and one prune call. The assertions must include:

```python
assert list(audio_crud.get_audio_files_bulk(session, requested)) == list(dict.fromkeys(requested))
assert store.downloaded_paths == [shared_pack.path]
assert counters.commits == 1
assert counters.prunes == 1
```

- [ ] **Step 2: Verify red.** Run `nix develop --command python /tmp/test_persistence_batching.py audio-pack`; expect import failure for `get_audio_files_bulk` or N+1 counter assertions to fail.
- [ ] **Step 3: Implement one-query row loading.** Deduplicate IDs with `list(dict.fromkeys(...))`; query all joined `AudioFile` rows once; calculate `missing = set(ids) - rows.keys()` and raise `KeyError(f"Audio files not found: {sorted(...)}")` before storage access.
- [ ] **Step 4: Route bulk pack operations through the loaded map.** Group downloads by `bucket_file_id`, slice bytes per requested ID, aggregate `_decrease_used_bytes` in memory, call `session.delete` for all rows, and commit once. Remove per-item `one(...)` calls and per-result `refresh(...)` loops where `expire_on_commit=False` already keeps values.
- [ ] **Step 5: Make single audio wrappers delegate to bulk.** `get_audio_file`, `read_audio_file`, and `delete_audio_file` call the corresponding bulk operation with one ID and unwrap the result. `bulk_delete_audio_files` resolves storage once and calls prune once after deletion.
- [ ] **Step 6: Verify green.** Re-run the temporary test mode and `nix develop --command python -m compileall -q src/shared/db/audio`; expect all assertions and compilation to pass.
- [ ] **Step 7: Commit.** Commit `audio/pack_crud.py` and `audio/crud.py` with `perf: batch packed audio persistence`.

### Task 2: Bulk waveform cleanup and atomic audio deletion

**Files:**
- Modify: `src/shared/db/waveforms/crud.py`
- Modify: `src/shared/db/audio/crud.py`
- Test: `/tmp/test_persistence_batching.py`

**Interfaces:**
- Produce `bulk_delete_waveforms(session: Session, audio_file_ids: Sequence[UUID], commit: bool = True) -> None`.
- `delete_waveform(...)` delegates to `bulk_delete_waveforms(..., [audio_file_id], commit=commit)`.
- `bulk_delete_audio_files` invokes waveform deletion with `commit=False`, packed deletion with `commit=False`, then commits once before pruning.

- [ ] **Step 1: Extend the temporary test** with waveforms in two waveform packs and audio in two audio packs. Assert deletion aggregates `used_bytes` correctly, removes all requested rows, performs one database commit, and invokes audio pruning once.
- [ ] **Step 2: Verify red.** Run `nix develop --command python /tmp/test_persistence_batching.py delete`; expect missing `bulk_delete_waveforms` or multiple commits.
- [ ] **Step 3: Implement bulk waveform deletion.** Select all `AudioWaveform` rows with joined packs in one query, group byte reductions by pack, assert every resulting `used_bytes >= 0`, delete loaded waveform rows, and honor the single `commit` flag.
- [ ] **Step 4: Make audio deletion one transaction.** Add `commit: bool = True` to internal packed deletion, perform waveform and packed mutations without intermediate commits, commit once in `bulk_delete_audio_files`, then prune once. Do not require every audio to have a waveform.
- [ ] **Step 5: Route bulk audio byte updates through bulk waveform cleanup.** Replace per-ID `delete_waveform(..., commit=False)` calls with one `bulk_delete_waveforms(..., commit=False)` call and retain one update commit/prune boundary.
- [ ] **Step 6: Verify green and compile.** Run the delete test mode and `nix develop --command python -m compileall -q src/shared/db/waveforms src/shared/db/audio`; expect success.
- [ ] **Step 7: Commit.** Commit the waveform/audio changes with `perf: batch waveform cleanup`.

### Task 3: High-volume relational CRUD

**Files:**
- Modify: `src/shared/db/datasets/crud.py`
- Modify: `src/shared/db/statistics/crud.py`
- Modify: `src/shared/db/voices/crud.py`
- Test: `/tmp/test_persistence_batching.py`

**Interfaces:**
- Produce `bulk_remove_audio_files_from_dataset(session, dataset_id, audio_file_ids) -> None`.
- Produce `bulk_create_statistics_entries(session, payloads: Sequence[StatisticsEntryCreate]) -> list[StatisticsEntry]`.
- Produce `get_voices_by_names(session, names: Sequence[str]) -> dict[str, Voice]` and `bulk_create_voices(session, payloads: Sequence[VoiceCreate]) -> list[Voice]`.

- [ ] **Step 1: Extend temporary tests** to assert one dataset membership delete statement, ordered statistics results, deduplicated voice names, and one commit per mutation call.
- [ ] **Step 2: Verify red.** Run `nix develop --command python /tmp/test_persistence_batching.py relational`; expect missing functions.
- [ ] **Step 3: Implement bulk membership removal.** Validate the dataset once, deduplicate audio IDs, execute `delete(dataset_audio_files).where(dataset_id == ..., audio_file_id.in_(ids))`, and commit once. The singular remove wrapper delegates and reloads only when its API must return a `Dataset`.
- [ ] **Step 4: Implement statistics bulk creation.** Convert each payload's `metadata` to `metadata_`, construct all `StatisticsEntry` objects, `add_all`, commit once, and return the same ordered list without refresh queries.
- [ ] **Step 5: Implement voice collection operations.** Query requested names with one `IN` statement, create deduplicated missing names with one `add_all`/commit, and preserve payload order. A unique-name conflict must surface explicitly rather than silently skipping.
- [ ] **Step 6: Verify green and compile.** Run the relational mode and compile the three CRUD packages through Nix.
- [ ] **Step 7: Commit.** Commit with `perf: batch high-volume relational writes`.

### Task 4: Bulk dataset, voice, delete, load, update, and statistics nodes

**Files:**
- Modify: `src/runner/nodes/dataset_writeback/nodes.py`
- Modify: `src/runner/nodes/audio_io/nodes.py`
- Modify: `src/runner/nodes/audio_segments/writeback.py`
- Modify: `src/runner/nodes/statistics/writeback.py`
- Test: `/tmp/test_persistence_batching.py`

**Interfaces:**
- Nodes call the Task 1-3 bulk facades exactly once per `execute(batch, context)`.
- `bulk_update_audio_files(session, payloads) -> dict[UUID, AudioFile]` remains the byte/metadata update interface.

- [ ] **Step 1: Add node-call-count tests** with patched bulk facades and two or more inputs. Assert each node makes one bulk call, supplies IDs/payloads in input order, and returns one output per input.
- [ ] **Step 2: Verify red.** Run `nix develop --command python /tmp/test_persistence_batching.py nodes`; expect single-item facade calls or multiple sessions.
- [ ] **Step 3: Convert membership and deletion nodes.** Collect audio IDs while checking cancellation, call bulk remove/delete once, then construct ordered results.
- [ ] **Step 4: Convert voice assignment.** Resolve the configured voice once, bulk-load audio rows, construct every `AudioUpdate`, call `bulk_update_audio_files` once, and rebuild outputs from the returned ID map.
- [ ] **Step 5: Convert audio loading and byte updates.** `LoadAudio` requests only missing byte IDs in one `bulk_read_audio_files` call; `UpdateAudioRecordBytes` bulk-loads metadata, builds all payloads, and calls `bulk_update_audio_files` once.
- [ ] **Step 6: Convert statistics writeback.** Build typed create payloads for the entire batch, open one session, call `bulk_create_statistics_entries`, and map model dumps back in order.
- [ ] **Step 7: Verify green and compile.** Run node mode and `nix develop --command python -m compileall -q src/runner/nodes`; expect success.
- [ ] **Step 8: Commit.** Commit node changes with `perf: batch persistence nodes`.

### Task 5: Batch artifacts, split persistence, and diarization voices

**Files:**
- Modify: `src/shared/db/assets/crud.py`
- Modify: `src/runner/nodes/audio_io/nodes.py`
- Modify: `src/runner/nodes/audio_segments/extract.py`
- Modify: `src/runner/nodes/audio_segments/speaker_split.py`
- Test: `/tmp/test_persistence_batching.py`

**Interfaces:**
- Produce `bulk_create_extra_files(session, payloads: Sequence[ExtraFileCreate]) -> list[ExtraFile]`.
- Split nodes use `get_audio_files_bulk`, `bulk_read_audio_files`, `bulk_create_audio_files`, `bulk_replace_audio_segments`, bulk membership, and bulk source deletion.

- [ ] **Step 1: Extend tests** for ordered bulk artifact results, one object-store construction, staged-upload cleanup on database failure, deduplicated source reads, one bulk split create, and one bulk voice create for all missing speaker names.
- [ ] **Step 2: Verify red.** Run `nix develop --command python /tmp/test_persistence_batching.py composite`; expect missing bulk calls.
- [ ] **Step 3: Implement bulk artifact creation.** Resolve the object store once, precompute IDs/paths/hashes, upload staged objects, add all rows and commit once, and delete only newly staged paths if commit raises. Make singular creation delegate and unwrap.
- [ ] **Step 4: Convert artifact node.** Build all `ExtraFileCreate` payloads and output metadata first, call `bulk_create_extra_files` once, then zip artifacts and source data strictly.
- [ ] **Step 5: Convert split extraction.** Deduplicate source IDs, bulk-load rows and bytes, then extract outputs in original input order.
- [ ] **Step 6: Convert split persistence.** Build all audio creates, create once, replace segments once, group dataset membership additions, and deduplicate final source removals/deletes. Validate replace-all completion metadata before mutation.
- [ ] **Step 7: Convert diarization voice persistence.** Collect all deterministic voice names after batched diarization, load them once, bulk-create missing names once, and pass the lookup map into clip construction without CRUD inside audio/speaker loops.
- [ ] **Step 8: Verify green and compile.** Run composite mode and compile touched packages through Nix.
- [ ] **Step 9: Commit.** Commit with `perf: batch composite audio persistence`.

### Task 6: Static audit and real graph verification

**Files:**
- Temporary: `/tmp/test_persistence_batching.py`
- Temporary: `/tmp/delete_audio_records_graph.json`
- No committed fixtures or tests.

- [ ] **Step 1: Run the complete temporary suite.** Execute `nix develop --command python /tmp/test_persistence_batching.py all`; expect all query, commit, prune, ordering, and node-call-count assertions to pass.
- [ ] **Step 2: Run static audits.** Use `rg -n -C 5 'for inputs in batch'` across registered node files and `rg -n 'session.commit\(\)'` across bulk CRUD. Confirm no high-volume persistence dependency call or commit remains inside a per-input loop.
- [ ] **Step 3: Run compilation and existing checks.** Run `nix develop --command python -m compileall -q src/shared src/backend src/runner src/runflow`; run the repository's existing focused check command discovered from `pyproject.toml`; expect zero errors.
- [ ] **Step 4: Restart only the shared stack.** Use `nix develop --command runflow-dev-stop`, `nix develop --command runflow-dev-session`, detach, then `nix develop --command runflow-dev-status`; expect one running `runflow-dev` session and `GET /health` to return `{"status":"ok"}`.
- [ ] **Step 5: Submit a real deletion graph.** Create audio fixtures through public CRUD, include multiple records sharing audio/waveform packs, submit a graph containing `TestingRunInput`/typed audio feeding `DeleteAudioRecords` through `POST /graphs/runs`, and save the request only in `/tmp/delete_audio_records_graph.json`.
- [ ] **Step 6: Inspect supported results.** Run `nix develop --command python -m cli runs`, `logs <run_id>`, and `failed <run_id>` if needed. Expect terminal success, every target record absent, pack `used_bytes` correct, and instrumentation showing one delete transaction and one prune call for the node batch.
- [ ] **Step 7: Smoke existing workflows.** Run the practical persistence portions of `workflows/audio_dataset_prep.json`, `workflows/deduplicate_overlapping_segments.json`, and `workflows/whisperx_merge_alignment.json`; expect unchanged output shape and lineage.
- [ ] **Step 8: Remove temporary artifacts.** Delete the two `/tmp` files and any temporary database fixtures through public CRUD.
- [ ] **Step 9: Review scope.** Run `git status --short`, `git diff --check`, file/folder line counts, and a final grep audit. Confirm no generated artifacts, caches, or unrelated changes are included.

## Completion Criteria

- `DeleteAudioRecords` performs one bulk waveform/audio transaction and one prune pass per scheduler batch.
- High-volume audio reads download each unique pack once per call.
- Audio, segment, waveform, statistics, voice, artifact, and membership mutations use one commit per bounded batch.
- Audited persistence nodes call collection facades once and preserve ordered output shape and lineage.
- Low-volume administrative CRUD remains singular.
- Temporary verification, compilation, shared-stack health, and real graph runs pass.
