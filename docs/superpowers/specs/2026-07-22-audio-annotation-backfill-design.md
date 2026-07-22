# Audio Annotation Backfill Design

## Goal

Normalize `style_prompt`, `voice_prompt`, `score`, and transcription `accuracy`
for every audio row using metadata already stored in PostgreSQL. Existing values
may be replaced when the derived value is more readable or materially richer.

The backfill must not read audio bytes and must not modify metadata, segments,
transcripts, alignment, audio storage, dataset membership, language, or speaker
identity.

## Derivation rules

- Build style prompts from readable emotion, intensity, delivery, noise/effects,
  and speaking-style fields described by `imports/order.md`.
- Build voice prompts from audible age, gender, accent or dialect, pitch, timbre,
  resonance, breathiness, range, roughness, articulation, and voice-description
  fields.
- Do not place technical provenance, filenames, speaker identifiers, political
  affiliation, transcript text, or opaque source codes in prompts.
- Decode enumerated values through the owning dataset importer or documented
  dataset mapping. Do not interpret the same numeric code globally.
- Prefer an audio/file/sample/overall MOS for `score`. Use a system or condition
  aggregate only when it is explicitly the dataset's per-audio rating.
- Populate `accuracy` only from a transcription accuracy, confidence, or
  equivalent recognition metric. Leave it `None` when no such field exists.
- Preserve a stronger existing value when metadata cannot derive a better one.

## Implementation

Use a temporary, typed backfill program run through Nix. It will inspect every
row through shared database facades, select a dataset-aware mapping, and produce
an immutable update proposal containing the row ID and destination fields only.
Metadata is an input to derivation but is never included in an update payload.

The program runs in two modes:

1. Dry run: report counts by dataset and field, before/after examples, unresolved
   opaque values, and mapping conflicts without committing.
2. Apply: batch updates through the shared audio CRUD layer after the dry run is
   reviewed. Each batch changes only the four destination columns.

## Safety and validation

- Assert before applying that every proposal leaves metadata and segments out of
  its writable representation.
- Reject non-finite scores and accuracy values.
- Reject unresolved numeric-only prompts.
- Verify representative mappings against each dataset's import code.
- After applying, re-query the entire table and compare metadata and segment
  fingerprints captured before the update.
- Report updated counts, remaining empty fields, and unresolved cases. Empty
  transcripts are not treated as errors.
