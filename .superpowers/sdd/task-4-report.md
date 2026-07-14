# Task 4 Report: Parquet speaker embedding shards

## Status

Implemented fixed-size float16 ECAPA Parquet shards and the lifecycle-managed
`ECAPASpeakerEmbed` node. The node duration-sorts and bounds inference by item
count and audio seconds, rejects short/non-finite inputs into explicit quality
rows, creates exactly one embedding run lazily from source metadata, reuses that
run for every shard, persists through asset CRUD, and reports item/seconds
progress with cancellation checks.

The segment source now includes `dataset_id` alongside `source_segment_count`,
which gives the embedding node the complete run identity without collector-owned
run creation.

## TDD evidence

- RED: the focused temporary test initially failed because PyArrow was absent,
  then failed on the expected missing `speaker_clustering.shards` and
  `speaker_clustering.embed_node` modules.
- GREEN: `nix develop --command uv run --with pytest pytest
  tmp_tests/test_speaker_embedding_shards.py -q` passed `7 passed`.
- Coverage included fixed-size 192-vector float16 Parquet round trips, absence of
  raw audio columns, explicit rejected rows, count/seconds group bounds, resource
  and batch policies, source dataset identity, one-run reuse across batches, and
  too-short/non-finite rejection reasons.
- `git diff --check`, `nix develop --command python -m compileall -q src`, and
  `nix develop --command uv lock --check` passed.
- Temporary tests and generated caches were removed per repository policy.

## Concerns

PyArrow cannot round-trip null fixed-size-list values through Parquet, so rejected
rows store a zero-vector sentinel and must be filtered by their explicit
`quality="rejected"` value. Accepted rows are enforced to contain finite vectors;
rejected rows cannot carry caller-provided embeddings.

The repository's bare Nix `pytest` command resolves `/usr/local/bin/pytest`, whose
Python cannot see project virtual-environment packages. Verification therefore
used Nix-wrapped `uv run --with pytest`; application imports and compile checks
used the Nix-wrapped project Python. The real SpeechBrain/model/artifact graph
smoke remains part of Task 5, where registration, collection, and the workflow are
implemented.

## Review fixes

- Every bounded group writes one artifact and emits its shared idempotent shard
  reference once per contributing scheduler input, preserving each original
  `__input_index__` lineage association.
- Any audio whose duration exceeds `maximum_batch_seconds` is stored as an explicit
  rejected shard row and is never passed to ECAPA inference.
- Every execute item now validates `dataset_id` and `source_segment_count` against
  the batch identity or reused embedding run before run creation or shard storage.
- RED: `nix develop --command uv run --with pytest pytest
  tmp_tests/test_task4_review_regressions.py -q` failed `4 failed`, demonstrating
  the oversized singleton and both metadata fields reaching persistence.
- GREEN: the same focused command passed `4 passed`; `git diff --check` and
  `nix develop --command python -m compileall -q src` also passed.
- A follow-up RED run of `nix develop --command uv run --with pytest pytest
  tmp_tests/test_task4_lineage_duration_review.py -q` failed `2 failed`: grouped
  output returned only index `0`, and oversized audio reached run creation.
- GREEN: the same focused command passed `2 passed`; a fresh Nix compile check
  and `git diff --check` passed in the same verification chain.
- A scheduler-level regression graph with valid and oversized inputs failed RED
  at the duration preflight before shard storage. GREEN passed `1 passed`, proving
  accepted inference remained within the seconds bound, the oversized input was
  stored with `duration_exceeds_maximum_batch_seconds`, and shard outputs retained
  source input lineages 0 and 1.
- The temporary regression test was removed per repository policy.
