# Task 5 Report: Durable speaker embedding collection

## Status

Implemented the `CollectSpeakerEmbeddings` node, atomic shard registration and
exact-count sealing, speaker embedding node registration, and the ECAPA example
workflow. Duplicate shard references are idempotent before and after sealing;
unknown artifacts cannot be appended to terminal runs. The collector emits one
set reference per run lifecycle and can recover the reference from an already
sealed durable run without emitting again for later duplicates.

## TDD and verification

- RED: `nix develop --command uv run --with pytest pytest
  tmp_tests/test_speaker_embedding_collection.py -q` failed because the collector
  module did not exist. A second RED run failed because the three speaker nodes
  were absent from the runner registry.
- GREEN: the focused temporary suite passed `5 passed`. It covered duplicate and
  out-of-order refs, one sealed output, atomic register-and-seal, matching terminal
  duplicates, rejection of unknown terminal artifacts, and registry discovery.
- `nix develop --command python -m compileall -q src` passed.
- Workflow Pydantic validation, runner schema discovery, `git diff --check`, and
  the 300-line file limit passed.
- The temporary test source was removed per repository policy.

## Real graph

After an empty-source registration smoke, a temporary dataset was created through
shared CRUD with two stored 16 kHz WAV files and two segments per file. The graph
was submitted through `POST /graphs/runs` as
`speaker_embedding_populated_smoke_v2` and completed successfully with 66 events.

The real runner log records `SpeakerSegmentSource` emitting four items,
`ECAPASpeakerEmbed` processing all four in one batch and emitting four lineage-
preserving references, and `CollectSpeakerEmbeddings` consuming four references
while emitting exactly one sealed-set packet. Database inspection confirmed one
sealed run with `expected_count=4`, `stored_count=4`, one Parquet shard containing
four 192-dimensional rows, and a 3,673-byte stored artifact. The temporary audio,
dataset, run jobs, artifact, and fixture script were removed afterward.

The first populated attempt exposed an incompatibility between HyperPyYAML 1.2.2
and ruamel.yaml 0.19.1 (`Loader.max_depth` was absent). A focused RED test reproduced
the loader failure. Pinning the directly used compatibility range to
`ruamel-yaml>=0.17.28,<0.19` resolved ruamel.yaml 0.18.17; the focused test passed,
the single shared dev session was restarted, and the populated graph then passed.

## Command caveat

The brief's bare Nix `pytest` resolves a Python without project dependencies and
failed on missing `pydantic`. As in Task 4, tests used Nix-wrapped
`uv run --with pytest` so they executed against the project environment.
