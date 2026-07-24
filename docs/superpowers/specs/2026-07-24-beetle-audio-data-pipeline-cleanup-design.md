# Beetle and Audio CRUD Data Pipeline Cleanup

## Goal

Make Beetle's training and validation data paths easy to trace while simplifying
the entire shared audio persistence package. Preserve deterministic training,
resume behavior, distributed sharding, bounded prefetching, cache limits, batch
shapes, database transaction boundaries, and stored-audio behavior.

The cleanup covers:

- `src/runner/nodes/training/beetle/data/`
- `src/shared/db/audio/`
- Repository callers affected by the audio CRUD module consolidation

Other shared CRUD domains are out of scope.

## Design principles

- Keep a boundary only when it owns meaningful behavior, state, lifecycle,
  concurrency, persistence, or storage policy.
- Remove forwarding objects, single-implementation protocols, duplicate export
  surfaces, and adapters that only rename an operation.
- Preserve every `shared.db.audio.crud` symbol currently used under `src/`,
  including its name and signature.
- Keep Beetle concepts out of shared audio persistence.
- Preserve behavior rather than redesigning sampling, prefetching, validation,
  or storage.
- Keep every file below 300 lines and every folder below 16 files.
- Use structured records for data exchanged between meaningful stages.
- Do not add compatibility shims or fallback paths.

## Target architecture

### Beetle training data

The training path becomes:

```text
DatabaseSegmentIndex
  -> ContinuousBatchPlanner
  -> DatabaseBatchLoader
  -> TrainingDataPipeline
  -> training loop
```

`DatabaseSegmentIndex` owns the immutable database snapshot, eligibility report,
validation candidates, and data fingerprint.

`ContinuousBatchPlanner` owns deterministic permutations, cut selection,
embedding-group planning, distributed rank slicing, pending windows, and
resumable planner state.

`DatabaseBatchLoader` is the concrete loading boundary. It reads current segment
metadata and unique WAV ranges in bulk, verifies segment identity, invokes the
collator, and returns one `BeetleBatch`.

`TrainingDataPipeline` owns the producer thread, one-window queue, decoded-byte
reservations, cancellation polling, failure propagation, state commit after
consumption, restoration, and resource shutdown.

The following forwarding layers are removed:

- `DatabaseBatchSource`
- `SharedSegmentBulkLoader`
- `SharedClipBulkLoader`
- `PlannedBatchLoader`
- The current pass-through `DatabaseBatchLoader`

Planned, fetched, and final batch records remain only where they represent
genuinely different data states.

### Shared audio persistence

The shared dependency direction becomes:

```text
callers
  -> shared.db.audio.crud
  -> responsibility-owned audio modules
  -> SQLAlchemy models and object storage
```

`shared.db.audio.crud` remains the public function facade. It re-exports
functions from their owning modules without wrapping them. Internal audio
modules never import the facade.

`shared.db.audio.__init__` exports schemas and storage configuration types, not
a second copy of the CRUD function surface.

Beetle uses `audio.crud` for database operations and the generic audio range
reader for streaming clips. Shared audio code does not know about Beetle
indices, batches, planners, or training.

### Beetle validation

Validation keeps its required two-stage lifecycle:

1. Load and preprocess recordings before tokenizer setup.
2. Collate the prepared recordings after tokenizers are available.

The single-implementation `ValidationDatabase` and
`SharedValidationDatabase` pair is removed. `ValidationLoader` calls the audio
CRUD facade directly.

## Module structure

### Shared audio package

| Target module | Responsibility | Consolidates |
|---|---|---|
| `audio/crud.py` | Stable public function exports | Current facade surface |
| `audio/catalog.py` | Audio lookup, search, sorting, and dataset-scoped file references | Query portions of `crud.py`, `rows_crud.py`, `references_crud.py` |
| `audio/files.py` | Create, update, delete, external-file creation, and object-store resolution | Mutation portions of `crud.py`, `delete_crud.py`, `external_crud.py` |
| `audio/segments.py` | Segment reads and mutations | `segments_crud.py` |
| `audio/segment_catalog.py` | Paged typed segment references | `segment_references_crud.py` |
| `audio/annotations.py` | Score and speaker-assignment bulk mutations | `scores_crud.py`, `speaker_assignment_crud.py` |
| `audio/packed.py` | Packed-byte reads, writes, and accounting | `pack_crud.py` |
| `audio/maintenance.py` | Orphan cleanup, pruning, and repacking | `pack_cleanup.py`, `pack_prune.py`, `repack_crud.py` |
| `audio/pack_store.py` | Pack writer and storage interfaces | Existing storage boundary |
| `audio/ranges/` | WAV slicing, cache, and concurrent range reads | Existing range package with `bulk.py` and `types.py` folded into `reader.py` |

`models.py` and `schemas.py` retain their responsibilities.

This consolidation brings the root audio folder below the 16-file limit.
Implementation work must split a proposed target further only when required to
stay below 300 lines, and any split must represent a distinct responsibility.

### Beetle data package

- `pipeline.py` contains `TrainingDataPipeline`, the bounded producer, and the
  public construction function.
- `prefetch.py` is removed.
- `loader.py` contains the concrete `DatabaseBatchLoader` and fetched records.
- `source.py` is removed.
- `collate.py`, `index.py`, `sampling.py`, `embedding_sampling.py`, `cuts.py`,
  `audio.py`, and `records.py` retain their behavior-owned boundaries.
- Validation remains split into orchestration, records, and tensor collation to
  satisfy the file-size limit.
- `data/__init__.py` exports only symbols consumed by training/runtime modules.
  Internal sibling modules use direct imports.

## Training data flow

1. `DatabaseSegmentIndex.build()` pages through
   `audio_crud.list_segment_references_page()` and constructs the immutable
   snapshot and fingerprint.
2. `ContinuousBatchPlanner.next_window()` produces deterministic
   `PlannedBatch` values and the exact `PlannerState` after each batch.
3. `TrainingDataPipeline` estimates and reserves decoded bytes for the window.
4. The producer calls `DatabaseBatchLoader.load(planned)` once per batch.
5. The loader opens one database session, bulk-loads current segment JSON, and
   asks its long-lived `BulkWavReader` for all unique WAV ranges.
6. The loader verifies segment index and ID, constructs fetched records,
   collates them, and returns one `BeetleBatch`.
7. The consumer receives batches in the existing order.
8. `mark_consumed()` commits the state associated with the delivered batch and
   releases its decoded-byte reservation.
9. `close()` stops the producer and closes the WAV reader executor.

## Preserved behavior

The refactor must preserve:

- Seeds, permutations, cycle transitions, grouping, window ordering, and rank
  slicing.
- Every serialized `PlannerState` and `DataPipelineState` field and meaning.
- Fingerprint and world-size checks during construction and restoration.
- Tensor values, shapes, dtypes, padding, masks, ordering, and batch metadata.
- One-window queueing and the configured decoded-byte budget.
- Cancellation polling while waiting for producer output.
- State advancement only after `mark_consumed()`.
- Every `shared.db.audio.crud` symbol and signature currently used under `src/`.
- Existing transaction and commit boundaries.
- Bulk database and object-storage operations.
- Pack offsets, stale-byte accounting, pruning, repacking, range reads, local
  caching, and cache eviction.
- Full-recording validation reads and the validation preload/collate split.

Checkpoint artifacts written before the module move are not a compatibility
target. Newly written checkpoints preserve the same sampler fields, validation,
and resume behavior. The project does not retain an old module solely as a
pickle import alias.

## Failure behavior

- Missing audio files raise `KeyError`.
- Segment index or segment ID drift raises `ValueError`.
- Invalid packed-storage metadata fails before audio decoding.
- Loader and collator exceptions cross the producer boundary unchanged.
- Producer termination without a queued result raises an explicit runtime
  failure.
- Shutdown fails if the producer cannot stop within the existing timeout.
- Restoration is rejected while a batch is in flight.
- No data is silently skipped, substituted, or defaulted.

## Verification

The implementation uses temporary characterization checks because committed
tests are not requested. Temporary files are removed before completion.

Checks cover:

- Index fingerprint, eligibility report, training pools, and validation
  candidates for fixed references.
- Planned keys, cuts, embedding groups, shard allocation, and complete planner
  state for fixed seeds.
- Identical next batches after uninterrupted and restored execution.
- State advancement only after consumption.
- Decoded-byte accounting after consumption, restoration, failure, and
  shutdown.
- Unique bulk segment and WAV requests, segment-drift failures, clip ordering,
  and batch tensor/metadata equivalence.
- Resolution and signatures of all repository `audio_crud` calls.
- Packed create, read, part-read, update, delete, stale-byte, prune, repack, and
  cached-range behavior.
- Validation selection, preload, and collation equivalence.
- Python import compilation and repository file/folder limits.
- A short real Beetle run through `./nix/run-venv.sh` with the supported
  training entry point and available local services/data.

All project commands run through `./nix/run-venv.sh`.

## Implementation sequence

1. Characterize current audio CRUD and Beetle pipeline behavior.
2. Consolidate audio query and mutation modules behind the facade.
3. Consolidate packed-storage maintenance and range-reader internals.
4. Replace Beetle's source and adapter chain with the concrete batch loader.
5. Move bounded prefetch into the training pipeline.
6. Simplify validation adapters and package exports.
7. Repeat the characterization checks and run a short real training smoke.
8. Remove temporary checks and confirm that no compatibility modules or aliases
   remain.

Behavioral changes, new data features, new storage policies, and changes to
other CRUD domains are out of scope.
