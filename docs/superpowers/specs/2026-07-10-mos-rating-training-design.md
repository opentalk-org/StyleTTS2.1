# MOS Rating, Training, and Inference Design

## Goal

Add a MOS annotation screen that presents two playable audio files from user-selected datasets, records a score for each file and the preferred file, trains a MOS regressor from those ratings with `facebook/wav2vec2-xls-r-300m`, and provides a workflow node that overwrites stored audio scores with model predictions.

## Chosen approach

Persist both forms of supervision:

- `audio_files.score` remains the current scalar MOS value and is updated by every submitted rating.
- A dedicated comparison row records the dataset, both audio IDs, both submitted scores, and the preferred audio ID at rating time.

This preserves compatibility with the existing score editor while giving pairwise training an explicit preference target. Deriving preferences only from scalar scores would discard the user's direct comparison choice, while storing comparison history in audio metadata would make sampling, validation, and training queries inefficient and weakly structured.

## MOS annotation data model

Create a `mos_comparisons` table with:

- UUID primary key.
- `dataset_id` referencing the dataset from which the pair was sampled.
- `audio_a_id` and `audio_b_id` referencing two distinct audio files.
- `preferred_audio_id`, which must equal one of the two audio IDs.
- `score_a` and `score_b` as finite floating-point values.
- `previous_score_a` and `previous_score_b`, preserving the scalar scores that existed immediately before submission.
- A timezone-aware creation timestamp.

Audio and dataset deletion cascades to affected comparison rows. Indexes support dataset-scoped chronological reads used to build manifests. A new Alembic migration creates the table, and the model module is imported by the shared database connection so Alembic sees it.

The shared `src/shared/db/mos/` feature owns models, schemas, and CRUD operations. The CRUD layer samples pairs, validates membership and preference invariants, updates both audio scores, inserts the comparison, and commits those changes as one transaction.

## Pair sampling and rating API

Add a backend `/mos` router with two operations:

- `GET /mos/pair?dataset_id=<uuid>&dataset_id=<uuid>` selects one eligible dataset from the requested set, then returns two distinct, non-virtual audio files from that same dataset. Sampling within one dataset ensures every saved comparison can later be selected by a single training dataset. Dataset membership is validated and datasets with fewer than two eligible files are excluded.
- `POST /mos/ratings` accepts the sampled dataset ID, two audio IDs, two finite scores, and the preferred audio ID. It rejects identical audio IDs, a preferred ID outside the pair, or files that are not both members of the supplied dataset. On success it atomically saves the comparison and overwrites both current audio scores.
- `GET /mos/ratings` returns newest-first, dataset-filtered comparison history through offset pagination, including compact audio details and whether the row can be modified.
- `PATCH /mos/ratings/{id}` changes the two scores and preference for the newest comparison and overwrites both current audio scores.
- `DELETE /mos/ratings/{id}` undoes the newest comparison, restores its two `previous_score` values, and deletes the comparison atomically. Restricting edit/undo to the newest row makes restoration deterministic.

The pair response contains only annotation fields: ID, name, duration, current score, and speaker. Audio playback continues through the existing `/audio-files/{id}/content` endpoint.

Random selection must avoid loading or sorting an entire dataset in application memory. Dataset eligibility and indexed UUID-based selection stay inside the shared CRUD layer so lists with millions of rows remain viable.

## MOS annotation UI

Add `mos` as a top-level screen and sidebar item. The feature lives in `src/frontend/src/features/mos/` and separates API calls, TanStack Query hooks, state/logic, and rendering.

The screen provides:

- A dataset multi-selection control populated through the existing dataset query.
- Two side-by-side rating cards using the shared `WaveformPlayer` with the existing audio content URL.
- The same score semantics as the segment editor: finite numeric values with two decimal entry precision and three-decimal display formatting. Each draft is initialized from the current score when present, but both values are required for submission.
- One action on each card—`Choose A/B as better and save`—enabled only when both scores are finite. Clicking it records that card as preferred and submits immediately; ties are not part of this workflow.
- A server-paginated, virtualized comparison history. All rows are viewable; the newest row exposes inline change and undo actions.

Submitting saves the rating, invalidates audio score/history queries, and fetches another random pair. Editing or undoing the newest row refreshes the same caches. Playback or request failures surface through the existing toast feedback. Score parsing/formatting and the score input presentation are extracted into reusable audio UI helpers shared by the segment editor and MOS cards.

## Base checkpoint catalog

Add a catalog item named `Wav2Vec2 XLS-R 300M · MOS base` for `facebook/wav2vec2-xls-r-300m`. A `mos_models` catalog task downloads its Hugging Face snapshot through the existing checkpoint download path and registers it as checkpoint type `mos_base` with model ID metadata. The generic `CatalogDownload` and `SelectCheckpoint` nodes continue to resolve the checkpoint; no MOS-specific checkpoint reference datatype is introduced.

The Checkpoints UI groups the catalog item under Training assets and recognizes both `mos_base` and trained `mos_model` types and tones.

## Training workflow

Add a MOS tab to the existing Training screen. Its graph is:

`TrainingRunInput -> SelectTrainingDataset -> BuildMosTrainingManifest -> MosModelTraining`

`SelectCheckpoint` broadcasts the selected `mos_base` checkpoint to both manifest building and training. The form reuses dataset/checkpoint pickers, schema-driven setting fields, and the queue card. Settings include display name, validation comparison count, batch size, learning rate, epochs, dataloader workers, comparison-loss weight, and checkpoint interval.

`BuildMosTrainingManifest` queries comparison rows for the selected dataset, validates that both referenced audio files remain available, creates deterministic train/validation splits by comparison row, and writes JSONL manifests in a run-specific folder. Each line records both audio IDs, both scores, and which side is preferred. It returns the existing generic `TrainingManifestPort`; MOS details live in manifest metadata rather than a new port type.

At least two comparisons are required so both train and validation splits are non-empty. Audio is read lazily in batches through the shared audio CRUD facade rather than copied into an unbounded in-memory collection.

## Model and loss

The model uses the local `facebook/wav2vec2-xls-r-300m` processor and `Wav2Vec2Model` encoder. Audio is decoded to mono, resampled to 16 kHz, and processed in padded batches. Masked mean pooling over the final hidden state feeds a linear scalar regression head. The scalar output is unbounded because the current audio score field is unbounded.

For predictions `p_a` and `p_b`, submitted scores `s_a` and `s_b`, and preferred sign `y` (`+1` when A is preferred and `-1` when B is preferred), the batch loss is:

`MSE(p_a, s_a) + MSE(p_b, s_b) + comparison_weight * softplus(-y * (p_a - p_b))`

The `softplus` term is the numerically stable negative log-sigmoid Bradley-Terry loss. It supplies the requested logarithmic comparison objective without taking the logarithm of a raw score difference, which is undefined for non-positive differences.

Training fine-tunes the encoder and regression head, checks cancellation between batches and validation steps, and reports item/epoch progress. The output folder stores the processor, encoder weights, regression-head weights, and a MOS configuration file. Publishing creates a `mos_model` checkpoint through the existing training-result path.

## Inference and score writeback

Add a `PredictMosScore` node in the MOS node family. It accepts batched `AudioPort` items and a broadcast `CheckpointRefPort` restricted to `mos_model`. The node loads the processor, encoder, and head once per checkpoint lifecycle, processes the whole incoming batch, and releases accelerator resources during teardown.

For each input it emits the same audio item plus a JSON writeback result containing the audio ID and predicted score. It bulk-overwrites `audio_files.score` through the public shared audio CRUD facade. Long batches check cancellation between decode/inference chunks and report completed item counts.

A normal inference workflow is:

`AudioSource -> LoadAudio -> PredictMosScore`

The node preserves one output per input and does not add MOS assumptions to `runflow`.

## Registration and discovery

Register `BuildMosTrainingManifest`, `MosModelTraining`, and `PredictMosScore` in the runner registry and MOS package exports. Their settings, typed ports, batch policies, resource policies, queue sizes, and categories are exported automatically through the existing `/schema` path so the workflow editor and Training screen discover them.

## Error behavior

Failures are explicit and actionable:

- Pair requests fail when no selected dataset contains two eligible audio files.
- Rating submission fails when membership, distinctness, finite-score, or preference invariants are violated.
- Change or undo fails when the target is not the newest comparison.
- Manifest creation fails when the selected dataset has fewer than two valid comparisons.
- Training fails when checkpoint files or required manifest fields are missing.
- Inference fails when given a checkpoint other than `mos_model` or audio without loaded bytes.

No compatibility fallbacks or silent skips are introduced.

## Verification

Behavior is developed test-first with temporary repository-local tests, which are removed before completion to follow the repository rule against committed tests. Verification covers:

- Migration/model metadata and MOS CRUD invariants.
- Pair API sampling and atomic score/comparison persistence.
- Paginated history plus newest-comparison change and score-restoring undo.
- Manifest split and loss calculation behavior.
- Catalog and node schema registration.
- Frontend type checking and production build.
- A real graph smoke run for MOS inference through `POST /graphs/runs`, inspected with the CLI.
- A minimal real training graph when the base checkpoint and sufficient ratings are available; otherwise the registered graph is validated through schema construction and the missing external runtime prerequisite is reported explicitly.

## Scope boundaries

This feature does not add tie ratings, rater accounts, score aggregation policies, active-learning pair selection, or MOS concepts to `runflow`. Each rating intentionally overwrites the current scalar score while retaining comparison history for training. Only the newest comparison can be changed or undone; older history is immutable.
