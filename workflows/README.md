# Workflows

Every workflow definition (JSON) across StyleTTS2 Studio lives here. Each file is a
`WorkflowCreate` payload (`name`, `hidden`, `data`) that the backend stores as a
saved workflow — it then appears in the UI workflow list.

## Seeding the backend

Run the seeder to save every definition in this folder into the backend. It is
idempotent — a workflow whose `name` already exists is skipped, so it is safe to
re-run against an existing backend:

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

### `deduplicate_overlapping_segments.json`

Loads audio segments and drops overlapping duplicates (`min_overlap_ratio` 0.5).

### `ds_v1_sample_import.json` / `ds_v2_sample_import.json`

Import a sample row from a Hetzner storage-box `ds_v1` / `ds_v2` parquet file and
save it as an audio record (v2 also saves segments and creates voices).

### `voice_embedding_pca.json`

Runs the StyleTTS style encoder over voices and plots a PCA scatter coloured by
voice.
