# Beetle Random Validation Sampling Design

## Goal

Replace hand-picked validation audio IDs with deterministic random full-recording
sampling from the configured training dataset. The behavior must work for
datasets with different duration distributions and must keep validation metrics
comparable throughout a run.

## Configuration

`ValidationConfig` contains a positive `sample_count` instead of an explicit
`audio_file_ids` list. The default Beetle configuration and the active local
Stage 1 configuration request 16 validation recordings.

This is a greenfield configuration change. There is no compatibility path for
the explicit-ID form.

## Candidate selection

Validation candidates come from the already-built `DatabaseSegmentIndex`. This
avoids a second dataset scan and guarantees that the validation population is
scoped by the same dataset and optional audio-file selection as training.

Stage 1 candidates are stored, non-virtual recordings that passed the existing
Stage 1 index eligibility rules. Stage 2 and Stage 3 candidates must additionally
have a configured language, complete text and phonemes, and exactly one voice
across every segment in the recording. Their selected set must contain the voice
diversity required by the unchanged grouped validation objectives.

Candidate audio IDs are sorted before sampling. A dedicated seed derived from
`runtime.seed` and the stage drives sampling without replacement. The same
configuration, dataset fingerprint, and stage therefore produce the same ordered
validation set across clean starts and checkpoint resumes. Validation events do
not resample recordings.

If fewer eligible recordings exist than `sample_count`, preparation fails with
an error containing the stage, requested count, and available count. Conditional
stages likewise fail clearly when the dataset cannot provide their required
voice diversity.

## Full-recording behavior

The selected IDs continue through `ValidationLoader`, shared audio CRUD, and the
uncropped model reconstruction path. The loader reads each full stored WAV,
collation pads only to model alignment, and artifact rendering trims only that
padding. Losses and WAV artifacts therefore cover the complete recording rather
than the 9,600-sample adversarial training window.

## Alternatives considered

- Database-side random ordering was rejected because it is expensive for large
  datasets and does not provide stable cross-database reproducibility.
- Reservoir sampling during index construction was rejected because the index
  already materializes eligible records and stage-specific filtering would make
  the build path more complex.
- Resampling on every validation event was rejected because changing examples
  would add sampling noise to validation curves.

## Verification

Temporary tests will prove that selection is deterministic, seed-sensitive,
without replacement, stage-aware, and strict about insufficient candidates.
They will also prove that the selected full waveform length is independent of
`adversarial.segment_samples`. The local config will be parsed after replacing
the explicit IDs, and a real training restart or resume will verify that emitted
validation WAVs match the selected database durations.
