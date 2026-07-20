# Segment Score and Accuracy Design

## Goal

Replace the ambiguous segment `confidence` field with separate optional `score` and `accuracy` fields across runtime models, stored segment JSON, APIs, frontend state, and runner nodes.

## Contract

- `score: float | None` represents segment quality, including imported MOS scores.
- `accuracy: float | None` represents transcription, alignment, or model certainty.
- Segment `confidence` is removed without a compatibility alias or fallback.
- Whole-audio `Audio.confidence` remains unchanged because it is outside the segment contract.
- The `audio_files.score` column remains the canonical whole-file MOS score.

Segments remain stored inside `AudioFile.segments` JSONB rather than gaining relational columns. Shared Pydantic schemas and backend request/response schemas expose both fields explicitly so all persisted segment writes have the same shape.

## Data Flow

Hetzner `mos_score` imports populate segment `score`. ASR transcription and alignment nodes populate segment `accuracy`. Segment extraction, writeback, external-record handling, training manifests, frontend editing, and API serialization preserve both fields. Statistics overlap selection and consensus calculations use `accuracy`, because those calculations measure transcript certainty rather than audio quality.

When a segment becomes a standalone `Audio`, its segment `score` does not replace whole-audio confidence. Any existing fallback that needs transcription certainty uses `segment.accuracy`, then the parent `Audio.confidence` only where the whole-audio contract requires a non-optional value.

## Validation and Verification

Both fields are optional floats and retain the existing permissive numeric range because upstream MOS and model certainty scales are not globally normalized. Temporary contract tests will verify the red/green rename and JSON round trip, followed by Python compilation, runner registry loading, frontend production build, and an exhaustive scan for obsolete structured segment confidence references. Temporary tests are removed before completion.
