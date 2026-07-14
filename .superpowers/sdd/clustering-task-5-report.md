# Clustering Task 5 Review Fix Report

## Outcome

Fixed the Task 5 assignment-policy and artifact findings without changing the
concurrent Task 3, Task 4, or orchestration work.

## Changes

- Replaced valid-cosine sentinels with optional best score, second score, and
  margin values. Missing candidates have null cluster and score fields; a
  single candidate has a best score but null second score and margin.
- Made the corresponding Parquet score fields nullable and preserved nulls
  through a write/read round trip.
- Added a concrete nullable `rejection_reason` to decisions and assignment
  artifacts so quality rejection detail is not discarded.
- Added typed assignment writer results containing shard paths and explicit
  accepted, provisional-new, ambiguous, and rejected counts.
- Rejected non-finite candidate scores and best-cluster dispersion before
  policy evaluation.

## TDD Evidence

The temporary suite `tmp_tests/test_task5_review_temp.py` was removed after
verification.

Initial RED:

```text
nix develop --command uv run --with pytest pytest \
  tmp_tests/test_task5_review_temp.py -q
ImportError: cannot import name 'AssignmentOutcomeCounts'
```

Final GREEN:

```text
nix develop --command uv run --with pytest pytest \
  tmp_tests/test_task5_review_temp.py -q
8 passed, 2 warnings in 3.20s
```

The warnings are FAISS SWIG deprecation warnings emitted during package import.

## Verification

```text
nix develop --command uv run --with ruff ruff check \
  src/runner/nodes/speaker_clustering/cluster_runtime/assignment.py \
  src/runner/nodes/speaker_clustering/cluster_runtime/artifacts.py \
  tmp_tests/test_task5_review_temp.py
All checks passed!

nix develop --command python -m compileall -q \
  src/runner/nodes/speaker_clustering/cluster_runtime/assignment.py \
  src/runner/nodes/speaker_clustering/cluster_runtime/artifacts.py
exit 0
```

Both changed source files remain below 300 lines. The relevant folders remain
below the repository file-count limit.
