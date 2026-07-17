# Beetle training

Beetle trains as three standalone, continuously sampled stages. There are no
epochs and no validation pass. Logging, loss schedules, optimizer schedules,
and atomic checkpoints use optimizer steps.

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
models. Checkpoints contain model, discriminator, frozen helper, EMA,
optimizer, scheduler, scaler, accumulated gradient, sampler, loss-schedule, and
Python/NumPy/Torch RNG state. SIGINT and SIGTERM request cancellation at the
next exact state boundary and write a final atomic checkpoint.

## Runtime reports

Startup reports the complete inference parameter count after loading the custom
BERT. A BERT-base-shaped 178-token fixture measures 199,603,199 parameters,
49,603,199 above the configured 150M ceiling, so the local BERT must be smaller
to meet the target. The report excludes prompt TextEncoder, frozen helpers,
discriminators, and training-only heads. The latent-to-audio path is profiled
and must remain strictly below 15 GFLOPs per generated second.

One learned vector represents each configured language. Duration prediction and
latent flow receive that same vector together with phoneme, pooled phoneme,
style, voice, and pre/post text/audio conditions. Duration consumes the complete
set at phoneme rate through its existing linear input projection; latent flow
projects each source independently at latent rate for AdaLN and configured
concatenation layers. Per-source dropout decisions are shared between rates.

The reusable execution package depends only on callback protocols. A future
Runflow node can map cancellation, progress, and artifact callbacks to node
context while keeping model, data, optimizer, and exact-resume behavior intact.
