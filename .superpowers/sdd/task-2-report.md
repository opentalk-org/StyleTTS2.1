# Task 2 Report: Keyset-paged segment source

## Status

DONE_WITH_CONCERNS

## Implementation

- Added `shared.db.audio.segment_references_crud` with typed `SegmentCursor` and
  `SegmentReference` values.
- Added a dataset-scoped segment count using PostgreSQL
  `jsonb_array_length(audio_files.segments)` without loading segment IDs.
- Added a bounded page query using `jsonb_array_elements(... ) WITH ORDINALITY`
  and the composite keyset `(audio_file_id, segment_index)`.
- Re-exported the segment reference CRUD functions through the existing audio
  CRUD facade.
- Added `SpeakerSegmentSource`, an I/O-leased streaming input node that fetches
  at most 1,024 segment references per execution, bulk-reads only the unique
  stored audio files required by that page, checks cancellation, and reports
  cumulative segment progress.
- Each output contains a clipped WAV `Audio` with exactly one clip-local
  `AudioSegment`. Audio and segment IDs are stable across reruns. Metadata
  includes the stable source audio/segment identity, source segment index,
  original span, and `source_segment_count`.
- Added the speaker clustering package export. Registry wiring remains deferred
  to Task 5 exactly as specified by the implementation plan.

## TDD evidence

Temporary test: `tmp_tests/test_speaker_segment_source.py` (removed before
commit per repository policy).

RED command:

```text
nix develop --command uv run --with pytest pytest tmp_tests/test_speaker_segment_source.py -q
```

Expected RED result:

```text
ModuleNotFoundError: No module named 'runner.nodes.speaker_clustering'
```

GREEN command:

```text
nix develop --command uv run --with pytest --with pytest-asyncio pytest tmp_tests/test_speaker_segment_source.py -q
```

Fresh GREEN result before temporary-test removal:

```text
2 passed in 3.08s
```

The tests covered:

- 1,025 references crossing two query pages;
- query page limits never exceeding 1,024;
- page-local unique bulk audio reads;
- stable ordered segment identities;
- exactly one segment per output;
- `source_segment_count` metadata;
- clip-local timing and duration;
- cancellation and final progress reporting;
- generated PostgreSQL SQL containing dataset scope,
  `jsonb_array_elements ... WITH ORDINALITY`, and the composite cursor.

An empty `segments` JSONB array is skipped by query semantics because the
lateral set-returning function produces zero rows for that audio record.

## Verification

- `nix develop --command python -m compileall -q src` — passed.
- Ruff on the three added source modules — passed.
- Generated PostgreSQL statement inspected with literal binds; it contains:
  `(audio_files.id, ordinality - 1) > (cursor_audio_id, cursor_index)` and orders
  by `audio_files.id, segment_index` with `LIMIT 1024`.
- `git diff --check` — passed.
- Added implementation files are 130 lines or fewer; folder/file limits remain
  satisfied.

## Concerns

- The brief's literal `nix develop --command pytest ...` does not use the
  project virtual environment in this checkout (`/usr/local/bin/pytest` lacks
  project dependencies), while the project venv does not include pytest. The
  tests therefore ran through Nix-wrapped `uv run --with pytest
  --with pytest-asyncio`, preserving the required Nix boundary.
- Full-file Ruff reports pre-existing facade re-export warnings and a
  pre-existing undefined `waveform_crud` name in `shared/db/audio/crud.py`.
  Task-scoped added modules pass Ruff; unrelated facade issues were not changed.
- A real graph run is deferred because Task 2 explicitly does not register the
  node; registry wiring and example workflow submission are Task 5 deliverables.
- An independent reviewer could not be spawned because the shared agent thread
  limit was already occupied. A local requirement/diff review was performed.
