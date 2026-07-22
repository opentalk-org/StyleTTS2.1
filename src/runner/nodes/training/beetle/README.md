# Beetle training

Beetle trains the acoustic model, text conditioning, latent flow, and GAN
jointly in one finite run. There are no epochs. Loss and optimizer schedules,
validation, MLflow reporting, and checkpoints use optimizer steps.

## Required data and assets

`data.selection.dataset_id` must identify a PostgreSQL dataset containing
non-virtual audio, text, language, speaker metadata, and the alignment metadata
required for mid-sentence cuts. `sentence_probability: 1` disables
mid-sentence sampling when word boundaries are unavailable. The dataset must
also contain enough distinct voices and recordings for the configured GE2E
groups.

Set the real aligner checkpoint asset in `architecture.aligner`, and ensure
`architecture.phoneme.model_path` contains the local custom BERT and tokenizer.
The StyleTTS2 JDC pitch checkpoint is loaded from the bundled external
reference.

## Launch

Run from the repository root through Nix:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
nix develop --command python -m runner.nodes.training.beetle.scripts.train \
  --config src/runner/nodes/training/beetle/config/default.yaml \
  --output /data/beetle/training
```

For data-parallel training:

```bash
nix develop --command python -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --mixed_precision no \
  -m runner.nodes.training.beetle.scripts.train \
  --config src/runner/nodes/training/beetle/config/default.yaml \
  --output /data/beetle/training
```

Configured autocast is applied inside the trainer, hence Accelerate uses
`--mixed_precision no`. Add `--resume <checkpoint_folder>` to resume from a
completed optimizer-step checkpoint. Checkpoints contain the complete model,
discriminator, EMA, optimizer, scheduler, scaler, sampler, loss schedule,
reporting state, and per-rank random state. Partial gradients and mid-step
phases are never saved; cancellation or failure leaves the latest valid
checkpoint untouched.

## Sampling and acoustic geometry

The full conditional path sees the complete utterance and its masks. The
acoustic/GAN path always uses a 19,200-sample, 64-mel-frame crop: 0.8 seconds at
24 kHz. Shorter targets are right-padded and their crop begins at frame zero,
so the crop always contains their real prefix and cannot select only padding.
Longer targets use a deterministic random crop while their conditional losses
continue to cover the full sequence.

The planner gathers
`data.prefetch.window_size * training.batch_size * world_size` examples,
sorts them by duration, forms homogeneous global batches, deterministically
shuffles batch order, then shards each batch across ranks. The prefetcher keeps
one active and one standby window, giving two buffers of
`window_size * batch_size` examples per rank. A checkpoint advances the sampler
only through batches the trainer actually consumed, including exact pending
batch order within a window.

## Runtime reports

MLflow uses the `beetle_training` experiment. Reports include losses, learning
rates, AMP scales, gradient norms, throughput, ETA, queue occupancy, system
metrics, full-recording validation metrics, and validation artifacts. The
latent-to-audio path remains subject to the configured inference parameter and
GFLOPs budgets. `runtime.compile_frame_count` must match the derived contextual
encoder geometry; the default 0.8-second acoustic crop derives 196 input mel
frames.

Each validation step writes two complete reports beneath
`validation/training/step_<step>/`: `full/` contains end-to-end conditional
synthesis and `audio/` contains AudioEncoder posterior reconstruction. Both
branches contain ground-truth/prediction WAVs and latent, F0, N, mel, STFT
magnitude, phase, and alignment plots for direct comparison. Shared loss
metrics remain in the step-level `metrics.json` manifest.
