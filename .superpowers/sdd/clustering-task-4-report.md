# Task 4 report: conservative microcluster review fixes

## Result

- Consolidation now requires explicit prototype-neighbor candidates and only
  merges reciprocal prototype matches.
- Support thresholds count distinct contributing members independently on both
  cluster sides, preventing one bridge member from satisfying the threshold.
- Disk-backed label construction, normalization, finalization, flushes, SQLite
  merge iteration, and remapping operate in configured blocks with cancellation
  checks.
- Oversized and high-dispersion clusters retain suspicious prototype metadata
  and have their member labels invalidated to `-1` for downstream audit.
- Prototype block operations live in
  `cluster_runtime/prototype_blocks.py` so all changed files remain below 300
  lines.

## TDD evidence

The temporary regression suite initially failed five cases covering the missing
prototype-neighbor contract, repeated bridge-member support, invalid-label
handling, and cancellation boundaries. After integration stabilization it was
expanded to seven focused cases and passed:

```text
nix develop --command uv run --frozen --with pytest python -m pytest \
  tmp_tests/test_task4_review_fixes_temp.py -q
.......                                                                  [100%]
7 passed
```

The temporary test was removed before commit.

## SQLite cancellation follow-up

- Both consolidation APIs now require an explicit positive configured
  `block_rows`; empty edge graphs still scan and remap labels in those configured
  blocks.
- Support-row insertion uses bounded `executemany` chunks with cancellation
  checks between chunks.
- SQLite connections install a progress handler that calls `check_cancel` during
  long insert and `COUNT(DISTINCT)` VM work, and clear it in `finally` before
  closing.
- Interrupted SQLite insert and aggregation errors now call `check_cancel`
  outside the SQLite callback boundary. Runtime cancellation is re-raised with
  its original exception; an interrupted SQLite error is preserved when the
  runtime callback returns normally.

Focused follow-up evidence:

```text
nix develop --command uv run --frozen --with pytest python -m pytest \
  tmp_tests/test_task4_sqlite_cancellation_temp.py -q
.....                                                                    [100%]
5 passed
```

The follow-up suite also rechecked reciprocal prototype gating and distinct
member support, then was removed before commit.

The final interruption-translation regression ran separately and was removed:

```text
nix develop --command uv run --frozen --with pytest python -m pytest \
  task4_interrupt_translation_temp.py -q
...                                                                      [100%]
3 passed, 2 warnings in 3.05s
```

## Scope

Only Task 4 microcluster/prototype/support modules and this report are included.
Concurrent edge, assignment, node, pipeline, persistence, registry, database,
workflow, and smoke-script changes were preserved without staging.
