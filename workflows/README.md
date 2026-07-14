# Workflows

Every workflow definition (JSON) across StyleTTS2 Studio lives here. Each file is a
`WorkflowCreate` payload (`name`, `hidden`, `data`).

## Examples tab (folder-driven, live)

The **Examples** tab of the workflow library is served straight from this folder
by `GET /workflows/examples`, which re-reads the `*.json` files on every request.
Add, edit, or remove a file here and it shows up the next time the library is
opened — no backend or database restart, and no seeding step. Malformed files are
skipped rather than breaking the listing. Set `RUNFLOW_WORKFLOWS_DIR` to point the
endpoint at a different folder.

## Seeding the backend (Saved tab)

Examples are ephemeral (they are never written to the DB). To keep an editable
copy under the **Saved** tab, run the seeder to store every definition in this
folder as a saved workflow. It is idempotent — a workflow whose `name` already
exists is skipped, so it is safe to re-run against an existing backend:

```bash
python workflows/save_workflows.py
```

Set `RUNFLOW_BACKEND_URL` to target a non-default backend (default
`http://127.0.0.1:8001`, matching `BACKEND_HOST`/`BACKEND_PORT` in
`nix/runflow-dev.sh`).

To register a single definition by hand instead:

```bash
curl -X POST http://127.0.0.1:8001/workflows \
  -H 'Content-Type: application/json' \
  -d @workflows/audio_dataset_prep.json
```

## Definitions

### `audio_dataset_prep.json`

The audio dataset-preparation pipeline:

```
AudioSource → LoadAudio → VadDetect (coarse 45-600 s chunks) → DeepFilterNetDenoise
  → ParakeetTranscribe → DiarizeSplitSpeakers (sortformer diarization + a random
  voice per speaker + punctuation-bounded, normal-distributed 1..X s split)
  → SaveAudioRecord → SaveAudioSegments (parakeet)
  → WhisperTranscribe → PhonemizeSegments → SaveAudioSegments (whisper)
  → CanaryTranscribe → SaveAudioSegments (canary)
  → AddAudioToDataset
```

Each source recording is split into single-speaker clips (1..X s, normal-distributed,
cut only at punctuation) with parakeet/whisper/canary transcripts and phonemes,
then added to a dataset — ready for StyleTTS2 finetuning.

`launch_source` selects which audio to run on (here: a specific file); change it to
`{"kind": "all_audio"}` or `{"kind": "dataset_audio", "dataset_id": "..."}` to run
over more inputs. The `AudioSource` params and the `to_dataset` / checkpoint node
ids are environment-specific — repoint them at your own audio, dataset, and
catalog checkpoints before running elsewhere.

### `whisperx_merge_alignment.json`

```
                              ┌─ WhisperXAlign (ck A) ─┐
AudioSource → LoadAudio → LoadAudioSegments            MergeAlignment → SaveAudioSegments (replace)
                              └─ WhisperXAlign (ck B) ─┘
```

Loads an audio that already has segments, force-aligns each segment's words with
two different WhisperX checkpoints, merges the two per-word alignments into the
best combined set (pairing same words even when their timings drift, keeping the
higher-scored timing), and replaces the stored alignment. Both align branches read
the same `LoadAudioSegments` output, so their audios share a lineage and pair up in
`MergeAlignment`. Repoint the `AudioSource` / `launch_source` `audio_file_ids` and
the two `ResolveCheckpoint` `checkpoint_id`s (whisperx aligner checkpoints from the
catalog) before running.

### `deduplicate_overlapping_segments.json`

Loads audio segments, collapses overlapping duplicates (`min_overlap_ratio` 0.5),
and merges their word alignments without duplicating words when aligner timings drift.

### `ds_v1_sample_import.json` / `ds_v2_sample_import.json`

The ds_v1 workflow imports a long recording, merges its recording metadata with
the matching ds_v2 metadata rows, and stores every transcript variant plus
absolute Parakeet word alignment. The ds_v2 workflow discovers sorted metadata
CSVs, applies its offset and limit across them, and derives each processed
Parquet path from the selected metadata filename. With `import_audio` enabled it
validates and attaches the selected bytes; both workflows store audio and segments.

### `ds_v2_metadata_import.json`

Run the same unified ds_v2 source with `import_audio` disabled and the same
`SaveAudioRecord` node in external mode. It stores metadata and transcript
segments without downloading Parquet audio or writing object-store packs. Records
retain the derived processed-Parquet row location and remain virtual.

### `voice_embedding_pca.json`

Runs the StyleTTS style encoder over voices and plots a PCA scatter coloured by
voice.

### `smart_turn_predict.json`

Loads selected audio as a batch and classifies whether each item completes a
conversational turn with Smart Turn v3.2. Every input is preserved while the node
exposes typed `turn_complete` and `probability` outputs. Replace the example audio
UUID in both the source node and `launch_source` before running it locally.
