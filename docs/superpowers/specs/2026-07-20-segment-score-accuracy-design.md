# Segment Score and Accuracy Design

## Goal

Remove `confidence` from audio and segments, using `score` for quality and `accuracy` for segment transcription certainty across runtime models, stored JSON, APIs, frontend state, and runner nodes.

## Contract

- A shared frozen `AudioAnnotations` Pydantic value object defines `speaker_id`, `voice_id`, `score`, `accuracy`, and custom `metadata` once.
- `Audio`, `AudioRecordRef`, and `AudioSegment` contain `annotations: AudioAnnotations`; they do not redeclare annotation fields.
- A shared read-only mixin exposes `.speaker_id`, `.voice_id`, `.score`, `.accuracy`, and `.metadata` properties for concise node reads. Constructors and mutations use the composed `annotations` object.
- `score` represents audio or segment quality, including MOS scores. `accuracy` represents transcription, alignment, or model certainty.
- `confidence` is removed without a compatibility alias or fallback.
- The `audio_files.score` column remains the canonical whole-file MOS score.

Audio rows gain nullable `speaker_id`, `voice_id`, and `accuracy` columns beside the existing `score` and `metadata` columns. Segments remain stored inside `AudioFile.segments` JSONB with one nested `annotations` object. Shared database and backend schemas use the same `AudioAnnotations` model. Frontend contracts mirror that nested object.

## Data Flow

Sources, synthesis nodes, and database reads populate audio annotations. Hetzner `mos_score` imports populate audio and segment `score`. ASR transcription and alignment nodes populate segment `accuracy`. Segment extraction, writeback, external-record handling, training manifests, frontend editing, and API serialization preserve the complete annotation object. Statistics overlap selection and consensus calculations use `accuracy`, because those calculations measure transcript certainty rather than audio quality.

When a segment becomes a standalone `Audio`, its segment `score` becomes the derived audio score when present; otherwise the parent audio score is preserved. Transcript certainty remains on `segment.accuracy` and is never used as an audio quality score.

## Validation and Verification

Both fields are optional floats and retain the existing permissive numeric range because upstream MOS and model certainty scales are not globally normalized. Temporary contract tests will verify the red/green rename and JSON round trip, followed by Python compilation, runner registry loading, frontend production build, and an exhaustive scan for obsolete structured segment confidence references. Temporary tests are removed before completion.
