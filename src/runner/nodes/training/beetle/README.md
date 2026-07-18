# Beetle training

Beetle trains as three standalone, finite stages that continuously sample the
dataset in a circle. There are no epochs. Loss schedules, optimizer schedules,
validation, MLflow reporting, and atomic checkpoints use optimizer steps. Each
stage stops exactly at its configured `total_steps`.

## Required data and assets

The YAML `data.selection.dataset_id` must identify a PostgreSQL dataset. The
index reads segment references through shared CRUD and accepts packed,
non-virtual audio with 1–45 second target segments. Every row must have a
language present in the explicit ordered `architecture.language.values` list;
missing and unconfigured values are rejected before stage pools are built. The
configured order defines checkpoint-stable embedding IDs and supports
mixed-language batches. Stages 2 and 3 additionally require text, phonemes,
voice labels, aligned word boundaries, enough distinct
voices for the configured voice groups, and enough recordings for style groups.
Empty or ineligible data fails before any model is loaded.

`validation.audio_file_ids` is a required, non-empty, explicitly ordered list
of stored audio UUIDs. Validation uses each complete recording and all of its
segments in database order; missing, virtual, unreadable, or incomplete entries
are rejected instead of skipped. Stage 2/3 validation requires at least two
recordings with distinct voices because it evaluates the unchanged contrastive
and GE2E objectives. The nil UUID in the baseline YAML is an intentional
required-to-replace placeholder.

`architecture.phoneme.model_path` is a local Transformers directory containing
the custom BERT and `BertTokenizerFast` files. The only configured phoneme token
count is `architecture.phoneme_token_count`, initially 178. Loading is strictly
local and intentionally performs no separate BERT metadata check.

The aligner is a checkpoint-folder asset in PostgreSQL. Set
`architecture.aligner.checkpoint_asset_id` to its real UUID and
`checkpoint_filename` to the state file inside that folder. The zero UUID in
the baseline YAML is a required-to-replace placeholder. Shared asset CRUD
materializes the folder; the training code does not access S3 or caches itself.
The frozen StyleTTS2 JDC pitch checkpoint is pinned with the local reference
files.
Set the final dataset, BERT path, and aligner UUID before Stage 1 because later
stage dependency checkpoints require the same configuration and data-index
fingerprints.

## Launch

Run every command from the repository root through Nix:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

MLflow is required. Setup, metric submission, asynchronous completion, and
artifact failures are fatal; training never falls back to a no-op logger.

```bash
nix develop --command python -m runner.nodes.training.beetle.scripts.train_stage1 \
  --config src/runner/nodes/training/beetle/config/default.yaml \
  --output /data/beetle/stage1
```

Stage 2 initializes its frozen audio path from a completed Stage 1 checkpoint
folder:

```bash
nix develop --command python -m runner.nodes.training.beetle.scripts.train_stage2 \
  --config src/runner/nodes/training/beetle/config/default.yaml \
  --output /data/beetle/stage2 \
  --stage1-checkpoint /data/beetle/stage1/checkpoints/checkpoint_<id>
```

Stage 3 initializes both prior stages:

```bash
nix develop --command python -m runner.nodes.training.beetle.scripts.train_stage3 \
  --config src/runner/nodes/training/beetle/config/default.yaml \
  --output /data/beetle/stage3 \
  --stage1-checkpoint /data/beetle/stage1/checkpoints/checkpoint_<id> \
  --stage2-checkpoint /data/beetle/stage2/checkpoints/checkpoint_<id>
```

Add `--resume <checkpoint_folder>` to resume the same stage. Resume validates
the configuration, compact data-index fingerprint, and stage before allocating
models. Checkpoints contain model, Stage 3 discriminator, frozen helper, EMA,
optimizer, scheduler, scaler, accumulated gradient, sampler, loss-schedule, and
Python/NumPy/Torch RNG state. They also preserve the MLflow run ID, pending
optimizer observation, metric accumulation, timing, queue counters, and last
reported/validated steps. SIGINT and SIGTERM request cancellation at the next
exact state boundary, write an atomic checkpoint, and leave the MLflow run
active for resume. Normal completion performs mandatory final validation,
flushes all work, writes a final checkpoint, and marks the run `FINISHED`.

## Runtime reports

Startup reports the complete inference parameter count after loading the custom
BERT. A BERT-base-shaped 178-token fixture measures 199,603,199 parameters,
49,603,199 above the configured 150M ceiling, so the local BERT must be smaller
to meet the target. The report excludes prompt TextEncoder, frozen helpers,
discriminators, and training-only heads. The latent-to-audio path is profiled
and must remain strictly below 15 GFLOPs per generated second.

All stages log to the `beetle_training` experiment. A completed optimizer step
uses one asynchronous MLflow metric batch containing every training loss,
learning rates, pre-clipping optimizer/module gradient norms, AMP scales,
items/s, steps/s, elapsed time, exact ETA, foreground overhead percentages,
metric/artifact queue occupancy, and host/process/GPU metrics. The first step is
excluded from rate and ETA estimates. Validation aggregate and ordered
per-sample losses are included in the same step batch.

Validation disables augmentation and every conditioning dropout source and
restores model modes plus Python/NumPy/Torch RNG state byte-for-byte. Stage 1
evaluates posterior reconstruction without a discriminator. Stages 2 and 3 use
the latent-flow EMA for exactly one shortcut integration step and save the
alignment. Stage 3 alone evaluates and trains the current StyleTTS discriminator
families. Artifacts live under
`validation/<stage>/step_<step>/sample_<position>_<audio_id>/`: every stage
saves ground-truth audio, latent/F0/`N`/mel/STFT-magnitude/phase plots; Stage 1
saves `recon.wav`, while Stages 2/3 save `pred.wav` and `alignment.png`.

One learned vector represents each configured language. Duration prediction and
latent flow receive that same vector together with phoneme, pooled phoneme,
style, voice, and pre/post text/audio conditions. Duration consumes the complete
set at phoneme rate through its existing linear input projection; latent flow
projects each source independently at latent rate for AdaLN and configured
concatenation layers. Per-source dropout decisions are shared between rates.

The reusable execution package depends only on callback protocols. A future
Runflow node can map cancellation, progress, and artifact callbacks to node
context while keeping model, data, optimizer, and exact-resume behavior intact.
