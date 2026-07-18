# Beetle MLflow and Validation Design

## Purpose and superseded baseline clauses

Beetle training will have finite, optimizer-step-based stage runs, complete
MLflow reporting, deterministic validation, and exact checkpoint resume. This
design amends the continuous-execution and training-stage sections of
`2026-07-17-beetle-training-design.md` in three ways:

- every stage stops at its configured optimizer-step limit;
- every stage validates on an optimizer-step cadence and at its final step;
- discriminators and adversarial losses train in Stages 1 and 3, not Stage 2.

There is still no dataset-pass or epoch concept. Training samples the eligible
dataset pools in a deterministic cycle until the stage reaches its step limit.
This work remains inside the Beetle node family. It may copy and adapt the
asynchronous reporting pattern from StyleTTS3, but it must not import
StyleTTS3-owned implementation modules.

## Configuration contract

Each `StageConfig` has two required positive integers:

- `total_steps`: the final completed optimizer step for the stage;
- `validation_every_steps`: the validation cadence for that stage.

The top-level validation configuration contains one required, non-empty,
explicitly ordered `audio_file_ids` sequence. The same ordered sample set is
used for all stages. Validation does not infer a split, sample randomly, or
silently reduce this list.

Preflight resolves every configured audio ID through shared database and audio
CRUD. It rejects duplicate, missing, virtual, unreadable, or stage-incomplete
records with the offending ID and reason. Stage 1 requires readable stored
audio. Stages 2 and 3 also require the complete ordered transcript, phoneme,
language, and voice metadata needed by their conditioning and alignment paths.
Empty training data or an incomplete validation list fails before model setup.

The configuration continues to reject any field whose name contains `epoch`.
`runtime.log_every_steps` remains a console-progress cadence; it does not gate
MLflow step metrics.

## Runtime architecture

The reusable training loop owns finite-step scheduling and emits typed events.
Two Beetle-owned packages keep the new responsibilities separate:

- `training/reporting/` owns MLflow run lifecycle, metric batches, timing,
  system sampling, bounded queues, and background failure propagation;
- `training/validation/` owns fixed-sample loading, stage evaluators, aggregate
  metrics, rendering, WAV serialization, and artifact manifests.

Stage trainers expose validation inputs and inference operations through typed
Beetle interfaces. Reporting and validation do not reach into CLI objects, and
the CLI only composes the trainer, callbacks, checkpoint manager, reporter, and
validator. Files remain below 300 lines and folders below 16 files.

GPU validation is synchronous with training because it uses the same models and
device. CPU conversion, plotting, WAV/JSON writing, and MLflow upload run behind
a bounded worker queue. Metric submission uses MLflow asynchronous operations
behind a separately bounded queue. A full queue applies backpressure instead of
growing without limit or dropping work.

## MLflow run lifecycle

MLflow is required. A fresh stage creates exactly one run using the repository's
configured tracking URI and records the resolved configuration. Stage 1,
Stage 2, and Stage 3 never share a run. The explicit MLflow run ID is stored in
every checkpoint. Resume reconnects to that exact run without using
process-global active-run state; a missing or incompatible run is an actionable
failure rather than a no-op fallback.

One asynchronous metric batch is submitted after every completed optimizer
step. Pending metric operations and artifact jobs are bounded. Background
exceptions are retained and raised on the training thread at the next reporting,
validation, or checkpoint boundary. Before each checkpoint, all metric
operations through that step are flushed and background errors are surfaced.

Normal completion performs any required final validation, writes a final
checkpoint, flushes all queues, and terminates the MLflow run as `FINISHED`.
Graceful cancellation flushes completed reporting work, writes an exact-resume
checkpoint, and leaves the MLflow run active for resume. A fatal runtime or
reporting error first attempts an emergency checkpoint and then attempts to mark
the run `FAILED`; the original error remains the reported failure.

## Per-step metric contract

All scalar loss components produced by the stage are logged, including raw
component losses, their weighted values, and combined totals. For gradient
accumulation, each component is averaged across the microsteps that contributed
to the completed optimizer step. The accumulator is checkpoint state so a
graceful mid-accumulation resume cannot lose or double-count completed
microsteps. Inactive objectives are absent rather than emitted as invented zero
losses.

Metric namespaces are stable:

- `train/loss/*`: every stage loss and total;
- `optimizer/*`: learning rates, scheduled loss weights, and AMP scale;
- `gradient_norm/*`: global optimizer and important-module pre-clipping norms;
- `performance/items_per_second` and `performance/steps_per_second`;
- `performance/elapsed_seconds`, `performance/eta_seconds`, and
  `performance/eta_hours`;
- `overhead/*`: measured foreground time shares and reporting queue state;
- `system/*`: host, process, and accelerator telemetry.

The first completed optimizer step establishes the timing origin and is excluded
from rates, elapsed time, ETA, and overhead percentages. Thereafter, elapsed
time includes data wait, forward/backward, optimizer work, validation,
checkpointing, reporting enqueue/backpressure, and other foreground work. It
does not include time while the training process is stopped. Timing totals and
the number of measured steps/items are checkpointed.

For measured steps:

```text
average_step_seconds = elapsed_seconds / measured_steps
eta_seconds = average_step_seconds * total_steps - elapsed_seconds
steps_per_second = measured_steps / elapsed_seconds
items_per_second = measured_items / elapsed_seconds
```

`measured_items` is the actual number of dataset items consumed across all
accumulation microsteps, not configured batch size multiplied by a counter.
ETA is clamped to zero only at completion.

Foreground timers partition elapsed time into data wait, compute/optimizer,
validation, checkpoint, reporting enqueue/backpressure, and residual time.
Each `overhead/<name>_percent` is `100 * category_seconds / elapsed_seconds`.
The report also includes pending metric operations, pending artifact jobs, and
queue-capacity utilization so asynchronous work is visible even when it does
not block training.

System telemetry follows the StyleTTS3 coverage: CPU utilization, system memory
utilization, process RSS, GPU utilization, GPU memory-controller utilization,
GPU memory used, GPU temperature, and GPU power. It is sampled without a second
MLflow call and included in the completed-step metric batch.

Gradient norms are measured before clipping. Every optimizer logs its global
norm. Important module groups are:

- Stage 1: AudioEncoder, FeatureLinear, Decoder, Generator, and discriminators;
- Stage 2: phoneme encoders, context encoders, conditioning projections,
  StyleEncoder, VoiceEncoder, DurationPredictor, and LatentFlowModel;
- Stage 3: all Stage 1 and Stage 2 groups.

## Validation execution

Validation runs after a completed optimizer step when the step is divisible by
that stage's cadence. It also runs at `total_steps` when that step was not
already validated. When validation and checkpoint cadence coincide, validation
finishes first so the checkpoint records it. Resume stores
`last_validated_step`; an interrupted validation is not marked complete and is
rerun deterministically.

Validation loads each configured audio ID in explicit order as a full recording,
uses its complete stored segments in database order, disables augmentation and
conditioning dropout, and uses fixed validation seeds. The evaluator saves and
restores module train/eval modes and all training RNG state. Validation-specific
generators isolate any sampling noise from training. Implementations may process
a recording in bounded windows, but every valid frame remains represented in
the per-sample manifest losses and saved full-recording audio.

Stage behavior is:

- Stage 1 evaluates posterior reconstruction with AudioEncoder, FeatureLinear,
  Decoder, Generator, and discriminators. It reports reconstruction, encoder
  KL, F0, `N`, discriminator, generator-adversarial, and feature-matching
  losses.
- Stage 2 evaluates duration likelihood, latent flow/shortcut objectives,
  alignment, style, voice, and consistency losses. Audio prediction uses the
  LatentFlowModel EMA weights with one shortcut step followed by the frozen
  Stage 1 latent-to-audio path.
- Stage 3 evaluates the combined acoustic, conditional, alignment, embedding,
  consistency, discriminator, adversarial, and feature-matching objectives.
  Conditional audio also uses the EMA LatentFlowModel with one shortcut step.

Only sample-count-weighted aggregate validation scalars are logged to MLflow
under `validation/*`. A JSON manifest records the ordered audio IDs, per-sample
losses, aggregate losses, seeds, stage, optimizer step, and artifact
paths.

## Validation artifacts

Artifacts use deterministic paths:

```text
validation/<stage>/step_<optimizer_step>/
  metrics.json
  sample_<one-based-position>/
```

All stages save `gt.wav`, a latent visualization, F0 and `N` plots, a paired
ground-truth/prediction mel plot, a paired STFT-magnitude spectrogram, and a
paired phase spectrogram. Stage 1 names generated audio `recon.wav`. Stages 2
and 3 name it `pred.wav` and additionally save the phoneme-to-frame alignment
matrix. Mel and multiresolution STFT values use the same StyleTTS2-compatible
mel-spectrogram loss implementation used for training; visualization code does
not introduce a second signal-processing definition.

Artifact rendering receives detached CPU tensors only. GPU evaluation can move
to the next validation sample after enqueueing bounded CPU work. A validation
event is complete only after its manifest and all artifact jobs have succeeded,
so `last_validated_step` never claims a partial upload.

## Exact resume and stage completion

The checkpoint payload is extended with MLflow run identity, accumulated timing,
loss accumulators, last reported step, last validated step, reporting queue
counters, and completion state. Existing model, optimizer, scheduler, scaler,
EMA, discriminator, gradient, RNG, sampler, and loss-schedule state remains.

At an optimizer boundary the order is optimizer completion, due validation,
submission of the single completed-step metric batch, metric flush when a
checkpoint is due, then checkpoint. This lets that step's timing metrics include
its validation work. At the final optimizer step the same order is followed even
when normal checkpoint cadence is not due. Resuming a final-step checkpoint only
performs missing finalization work; it never consumes another batch or updates
parameters beyond `total_steps`.

Exact resume means graceful interruption preserves every completed microstep,
accumulated gradient, loss accumulator, sampler position, and completed
optimizer step. MLflow uses deterministic metric keys and optimizer-step values;
re-emission after an abrupt crash may be at-least-once, but it cannot change the
training state or create a missing step. Abrupt process loss resumes from the
latest successfully written atomic checkpoint, as in the baseline design.

## Verification and failure behavior

Temporary verification, run through `nix develop --command`, covers strict
configuration, finite stopping, first-step timing exclusion, ETA arithmetic,
microstep loss averaging, actual item throughput, pre-clipping gradient norms,
bounded queue backpressure, background-error propagation, MLflow run resume,
final/cancel lifecycle, validation cadence, deterministic ordered artifacts,
mode/RNG restoration, and checkpoint resume during accumulation and validation.

The repository currently has no collected dataset. Verification may create
temporary synthetic database/audio records only through shared CRUD, exercise
the three stage paths with reduced models and short audio, and remove those
records and generated files afterward. No permanent sample fallback is added.
If a Beetle node adapter is introduced, its end-to-end check must use a real
registered graph; the current standalone scripts are verified through their
public CLI path. Temporary tests and scripts are removed before completion, in
accordance with repository policy.
