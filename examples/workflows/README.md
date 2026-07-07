# Example workflows

`POST` one of these definitions to the backend to register it (it then appears in
the UI workflow list):

```bash
curl -X POST http://127.0.0.1:8001/workflows \
  -H 'Content-Type: application/json' \
  -d "$(jq '{name, data, hidden: false}' examples/workflows/audio_dataset_prep.json)"
```

## `audio_dataset_prep.json`

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
