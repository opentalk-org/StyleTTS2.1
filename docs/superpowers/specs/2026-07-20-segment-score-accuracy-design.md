# Segment Score and Accuracy Design

## Goal

Remove `confidence` from audio and segments, using `score` for quality and `accuracy` for segment transcription certainty across runtime models, stored JSON, APIs, frontend state, and runner nodes.

## Contract

- `score: float | None` represents segment quality, including imported MOS scores.
- `accuracy: float | None` represents transcription, alignment, or model certainty.
- Segment `confidence` is removed without a compatibility alias or fallback.
- Runtime `Audio.confidence` is renamed to `Audio.score` without a compatibility alias or fallback.
- The `audio_files.score` column remains the canonical whole-file MOS score.

Segments remain stored inside `AudioFile.segments` JSONB rather than gaining relational columns. Shared Pydantic schemas and backend request/response schemas expose both fields explicitly so all persisted segment writes have the same shape.

## Data Flow

Sources, synthesis nodes, and database reads populate `Audio.score`. Hetzner `mos_score` imports populate audio and segment `score`. ASR transcription and alignment nodes populate segment `accuracy`. Segment extraction, writeback, external-record handling, training manifests, frontend editing, and API serialization preserve both segment fields. Statistics overlap selection and consensus calculations use `accuracy`, because those calculations measure transcript certainty rather than audio quality.

When a segment becomes a standalone `Audio`, its segment `score` becomes the derived audio score when present; otherwise the parent audio score is preserved. Transcript certainty remains on `segment.accuracy` and is never used as an audio quality score.

## Validation and Verification

Both fields are optional floats and retain the existing permissive numeric range because upstream MOS and model certainty scales are not globally normalized. Temporary contract tests will verify the red/green rename and JSON round trip, followed by Python compilation, runner registry loading, frontend production build, and an exhaustive scan for obsolete structured segment confidence references. Temporary tests are removed before completion.
