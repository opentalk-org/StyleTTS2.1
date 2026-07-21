# Speaker Identity Unification Design

## Goal

Remove the duplicate Voice identity from the backend, runner, shared schemas, and frontend. `speaker_id` becomes the only speaker identity used by stored audio, segments, workflow nodes, training data, APIs, and UI.

## Data model

`AudioFile.speaker_id` remains a nullable string. Audio and segment annotations retain `speaker_id` and remove `voice_id`. The empty `voices` table and the nullable `audio_files.voice_id` column are dropped by an Alembic migration; no data conversion is required because both currently contain zero values.

Speakers are derived from distinct non-null `audio_files.speaker_id` values. There is no separate speaker registry and no empty speaker record. Imported speaker IDs therefore appear immediately.

## Backend and shared CRUD

Replace `/voices` with `/speakers`. The speaker list returns a paginated, searchable row containing the speaker ID, audio count, segment count, and associated dataset IDs.

Renaming a speaker updates every matching audio row and the `speaker_id` inside each stored segment annotation in one shared CRUD operation. Deleting a speaker clears those same fields while retaining audio, dataset membership, packed bytes, and segments. Bulk filter operations use the same CRUD facade.

All callers access these operations through `src/shared/db/speakers/crud.py`; backend routes do not issue ad hoc persistence queries.

## Runner and workflows

Remove `voice_id` from shared annotation types, audio references, node payloads, speaker-clustering assignments, statistics, TTS inputs, training records, and workflow schemas. Where code currently assigns a Voice UUID, it assigns a string `speaker_id` instead. Existing speaker-oriented names remain unchanged.

Third-party model terminology such as a model's internal voice embedding or a provider's `voice` argument may remain when it does not represent the project's persisted identity. Paper/reference text is not rewritten.

## Frontend

Rename the Voices feature, route, navigation label, types, queries, components, and API calls to Speakers. The screen reads `/speakers`; rename and delete retain their current UX, while create is removed because a derived speaker cannot exist without audio.

Audio and segment selectors query Speakers instead of using `SPEAKER_NAMES`. User-facing identity labels use “speaker.” `voice_prompt` remains because it is descriptive conditioning text, not identity.

## Migration and compatibility

This greenfield project does not keep `/voices`, `voice_id`, or compatibility aliases. The migration drops the unused table and column directly. Existing imported `speaker_id` values remain intact.

## Verification

Temporary tests verify speaker aggregation, rename propagation, delete-to-clear behavior, and API responses. Static searches verify no project identity `voice_id` remains outside historical migrations or external reference material. Frontend type-check/build and Python compile checks must pass. A live API smoke test confirms imported speakers appear and a reversible rename/clear cycle works through the public endpoint.
