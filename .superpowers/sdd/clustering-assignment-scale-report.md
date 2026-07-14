# Speaker clustering assignment scale fix report

## Outcome

- Replaced corpus-sized in-memory established and prototype-neighbor arrays with
  block-built memmaps under the pipeline scratch directory.
- Kept stable prototype root IDs, exact reranking, deterministic iteration, and
  accepted-only prototype update semantics unchanged.
- Added cancellation checks throughout prototype selection, FAISS prototype
  training/add/search boundaries, assignment search, prototype neighbor scans,
  memmap flushes, and prototype Parquet shard scans/writes.
- Added explicit cleanup for selection and neighbor memmaps on cancellation and
  failure, with pipeline-owned mappings closed after their consumers finish.

## TDD evidence

The temporary regression initially failed during collection because the
disk-backed selection interface did not exist:

```text
nix develop --command uv run --frozen --with pytest python -m pytest \
  task_assignment_scale_temp.py -q
ImportError: cannot import name 'PrototypeSelection'
```

After implementation and cancellation coverage expansion:

```text
nix develop --command uv run --frozen --with pytest python -m pytest \
  task_assignment_scale_temp.py -q
.....                                                                    [100%]
5 passed, 2 warnings in 3.09s
```

The warnings are FAISS SWIG deprecation warnings emitted during import. The
temporary test was removed before commit.

## Integration verification

```text
nix develop --command uv run --frozen --with pytest python -m pytest \
  tmp_tests/test_task6_lifecycle_temp.py -q
...                                                                      [100%]
3 passed, 2 warnings in 3.54s

nix develop --command uv run --frozen --with ruff ruff check \
  src/runner/nodes/speaker_clustering/cluster_runtime/assignment_runtime.py \
  src/runner/nodes/speaker_clustering/cluster_runtime/artifacts.py \
  src/runner/nodes/speaker_clustering/cluster_runtime/prototype_blocks.py \
  src/runner/nodes/speaker_clustering/cluster_runtime/pipeline.py
All checks passed!
```

All changed source files remain below 300 lines and the runtime folder remains
within its file-count limit.
