# Canonical Speaker ID Design

## Goal

Use `speaker_id` as the only structured field for a speaker label throughout stored audio metadata, stored segments, runtime models, backend APIs, frontend state, runner nodes, and workflow configuration.

## Scope

- Rename audio and segment fields named `speaker` to `speaker_id`.
- Remove duplicate writes where both `speaker` and `speaker_id` carry the same value.
- Rename related API form fields, response properties, sort values, settings, and internal variables when they represent the structured label.
- Keep human-facing prose that uses the word “speaker.”
- Keep `voice_id`: it identifies a persisted voice record and is distinct from a source or diarization speaker label.
- Keep model-training terms such as `speaker_ids` when they already describe numeric class indices rather than serialized project fields.

## Data Contract

Audio metadata and segment dictionaries store an optional string under `speaker_id`. Runtime `AudioSegment` and `Transcript` objects expose `speaker_id`. Backend and frontend audio/segment contracts expose `speaker_id`, and upload requests send `speaker_id`. Code reads only `speaker_id`; no compatibility alias for `speaker`, `speakerName`, or `speakerId` is introduced because the project is greenfield.

Speaker IDs need not be UUIDs. They may be source-provided labels such as uploader IDs or diarization labels. `voice_id` remains an optional UUID beside `speaker_id`.

## Data Flow

Source and diarization nodes produce `speaker_id`. Segment extraction, persistence, transcription, statistics, clustering labels, manifests, backend serialization, and frontend editing pass that same key without translating it to a display-name field. Voice assignment may update both `speaker_id` and `voice_id`, because those fields represent different identities.

## Errors and Validation

Required segment payloads fail validation when `speaker_id` is absent. External model adapters may read a third-party model's `speaker` output at their boundary, but immediately map it into the internal `speaker_id` contract. No stored-data fallback reads obsolete keys.

## Verification

- Add temporary contract tests before implementation and observe them fail.
- Run focused Python tests and frontend type/build checks through `nix develop --command ...`.
- Run repository-wide searches proving structured `speaker`, `speakerName`, and `speakerId` fields are gone while allowed domain prose and external-library boundary keys remain.
- Remove temporary tests before completion, as required by repository policy.
