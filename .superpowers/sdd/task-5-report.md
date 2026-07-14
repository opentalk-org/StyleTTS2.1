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

The acceptance fixture used only public backend APIs: one temporary dataset, two
uploaded 16 kHz WAV files, and two segments per file. The first populated graph
failed consistently before inference because HyperPyYAML 1.2.2 accepts
`ruamel.yaml>=0.17.28`, while ruamel-yaml 0.19.1's `Loader` does not initialize the
`max_depth` attribute its composer reads. Package metadata and loader source
confirmed the boundary, and a minimal HyperPyYAML load reproduced RED. Constraining
ruamel-yaml to `>=0.17.28,<0.19` resolved 0.18.17 and made the same test pass.

After restarting the single shared dev session so its runner loaded the resolved
package, the identical graph succeeded as
`speaker_embedding_api_verify_20260714_c` with 66 events. CLI logs and the
persisted snapshot recorded:

- `SpeakerSegmentSource`: one input task and four output segments.
- `ECAPASpeakerEmbed`: one real batch with size/input/output `4/4/4`.
- `CollectSpeakerEmbeddings`: four deliveries of the same shard reference and
  output counts `1, 0, 0, 0`, proving idempotency and exactly one set output.
- Durable run `61a357a4-506f-48e7-8136-0040f4d8c9a2`: `sealed`, expected count 4,
  stored count 4, one shard, dimension 192, artifact size 4,787 bytes.
- Parquet result: four accepted rows in stable segment/label order,
  `fixed_size_list<halffloat>[192]`, with round-tripped norms from `0.999994` to
  `1.000030`.

Both uploads, the dataset, generated artifact, verification jobs, and local
WAV/Parquet files were deleted through the public APIs or local cleanup.

## Collector run binding

A focused temporary regression test first passed against the recovered working
tree, then failed against commit `1101ed1` because a second run ID was accepted.
Restoring the run-bound collector made both checks pass: mixed run IDs are
rejected and duplicate references for the bound run emit at most once. The test
was removed per repository policy.

## Command caveat

The brief's bare Nix `pytest` resolves a Python without project dependencies and
failed on missing `pydantic`. As in Task 4, tests used Nix-wrapped
`uv run --with pytest` so they executed against the project environment.
