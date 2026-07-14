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

`nix develop --command runflow-dev-status` confirmed the shared `runflow-dev`
session was already running. `workflows/speaker_embedding_ecapa.json` was submitted
through `POST /graphs/runs` as `speaker_embedding_task5_smoke`; the Nix-wrapped CLI
reported `succeeded` with four events.

The smoke dataset contains zero files (all datasets returned by `/datasets` also
reported zero files), so this run proves graph parsing, registration, dispatch,
and empty-source completion only. It could not exercise ECAPA batching or produce
a sealed shard manifest. `python -m cli logs` returned 404 because an empty source
created no node log. A populated segmented dataset and available model/GPU are
required for the requested multi-input artifact verification.

## Command caveat

The brief's bare Nix `pytest` resolves a Python without project dependencies and
failed on missing `pydantic`. As in Task 4, tests used Nix-wrapped
`uv run --with pytest` so they executed against the project environment.
