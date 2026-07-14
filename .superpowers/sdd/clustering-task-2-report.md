# Clustering Task 2 Review Fix Report

## Outcome

Fixed the review findings against `5dafcef` while leaving the concurrent Task 4
microcluster, prototype, and temporary-test files unchanged.

## Changes

- Replaced per-block `vstack`/concatenate sampling with a deterministic,
  fixed-capacity vector reservoir. Preallocated vectors and metadata are updated
  in place, while a bounded heap tracks the current worst retained priority.
- Added the exact Flat test profile through `FaissIndexSettings.for_test()`.
- Made the production defaults the usable canonical FAISS factory
  `IVF65536_HNSW32,Flat`, with 1,000,000 training rows, 64 probes, and seed 0.
  FAISS requires the trailing storage codec; `IVF65536_HNSW32` alone does not
  parse as a complete index factory string.
- Reconciled every `SpeakerEmbeddingSetRef` against speaker CRUD before reading:
  its run must be sealed, artifact IDs must exactly equal the ordered durable
  shard manifest, and manifest row totals must equal `item_count`.
- Kept artifact materialization behind asset CRUD and added cancellation checks
  immediately before every artifact resolution.
- Added cancellation checks immediately before and after FAISS training.

## TDD Evidence

The temporary suite was created at repository root rather than under
`tmp_tests/`, which was concurrently owned by Task 4, and removed before commit.

Initial RED:

```text
nix develop --command uv run --with pytest pytest task2_review_tests_temp.py -q
7 failed
```

The failures covered missing profiles, repeated retained-sample copies, missing
sealed/manifest/count validation, missing artifact-resolution cancellation, and
missing cancellation around training.

A separate RED exposed the incomplete production FAISS grammar:

```text
RuntimeError: could not parse index string IVF65536_HNSW32
```

Final GREEN before temporary-test removal:

```text
nix develop --command uv run --with pytest pytest task2_review_tests_temp.py -q
7 passed
```

## Verification

```text
nix develop --command uv run --with ruff ruff check \
  src/runner/nodes/speaker_clustering/faiss_index.py \
  src/runner/nodes/speaker_clustering/reservoir.py \
  src/runner/nodes/speaker_clustering/shard_reader.py
All checks passed!

nix develop --command python -m compileall -q \
  src/runner/nodes/speaker_clustering src/shared/db/speakers src/shared/db/assets
exit 0

git diff --check
exit 0
```

All changed source files remain below 300 lines.
