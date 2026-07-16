# iSTFTNet2-MB Vocoder Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a standalone five-epoch LJSpeech trainer for plain iSTFTNet2-MB with Wave-U-Net GAN, multi-resolution mel losses, full-audio validation media, and Aim logging.

**Architecture:** A small `vocoder_training` package separates backend audio caching and crop loading, GPU mel transforms, media/Aim reporting, and the optimization loop. A directly runnable CLI composes those units and exposes a bounded smoke mode before the real five-epoch run.

**Tech Stack:** Python 3.11, PyTorch, torchaudio, SoundFile, matplotlib, Aim, shared PostgreSQL/audio CRUD, Nix development shell.

## Global Constraints

- Train plain `ISTFTNet2MB` from scratch at 22.05 kHz on exact 8192-sample crops.
- Hold out 16 full utterances for normal validation and save/log every pair each epoch.
- Use LSGAN, feature matching weight 2, and mean three-resolution mel loss weight 45.
- Use Adam at `2e-4` with betas `(0.5, 0.9)` and five normal epochs.
- Access backend audio only through shared CRUD facades and keep backend-pack memory bounded.
- Keep every source file below 300 lines and do not retain committed tests.
- Run every Python/test/training command through `nix develop --command`.
- Inspect parameter counts only after behavioral verification.

---

### Task 1: Backend audio cache and crop-only DataLoader

**Files:**
- Create: `src/runner/nodes/training/styletts3/testing/vocoder_training/audio_data.py`
- Create temporarily: `/tmp/test_istftnet2_training.py`

**Interfaces:**
- Produces: `AudioEntry`, `AudioSplits`, `prepare_backend_audio(dataset_id, cache_dir, validation_samples, max_train_items)`, `CropDataset`, and `build_train_loader(entries, batch_size, workers)`.
- Consumes: shared `database_session`, audio CRUD paging, bucket locations, and bulk reads.

- [ ] **Step 1: Write failing temporary tests**

Create tests that write known WAV ramps, verify `CropDataset` returns exactly 8192 samples from files longer than 8192, verify short files are excluded by entry inspection, and verify `build_train_loader` has `drop_last`, pinning, and stable batch shapes.

- [ ] **Step 2: Verify RED**

Run: `nix develop --command python /tmp/test_istftnet2_training.py`

Expected: import failure because `vocoder_training.audio_data` does not exist.

- [ ] **Step 3: Implement bounded backend export and seeked crops**

Use paged `list_audio_file_references_page` to obtain deterministic dataset ids. Split validation before limiting smoke training items. Group selected ids by `audio_bucket_locations`; call `bulk_read_audio_files` once per group and write each WAV to the cache. Inspect with `soundfile.info`, enforce 22.05 kHz, exclude training entries shorter than 8192, and implement each `__getitem__` with `SoundFile.seek()` followed by exactly 8192 frames.

- [ ] **Step 4: Verify GREEN**

Run the temporary test and expect all data tests to pass.

### Task 2: Batched conditioning and multi-resolution mel loss

**Files:**
- Create: `src/runner/nodes/training/styletts3/testing/vocoder_training/mel.py`
- Modify temporarily: `/tmp/test_istftnet2_training.py`

**Interfaces:**
- Produces: `MelResolution`, `LogMelSpectrogram`, `MultiResolutionMelLoss`, `conditioning_mel()`, and `pad_to_hop()`.
- Consumes: batched float waveforms shaped `(B, T)`.

- [ ] **Step 1: Add failing geometry and loss tests**

Assert that the conditioning transform maps `(B, 8192)` to `(B, 80, 32)`, `ISTFTNet2MB` maps that conditioning back to `(B, 8192)`, identical inputs have zero multi-resolution loss, and perturbed inputs have positive finite loss.

- [ ] **Step 2: Verify RED**

Run the temporary test and expect missing mel interfaces.

- [ ] **Step 3: Implement GPU-vectorized log-mel modules**

Register Hann windows and Slaney mel filterbanks, use HiFi-GAN reflection padding and non-centered STFT, log-clamp energy at `1e-5`, and average the configured `512/128`, `1024/256`, and `2048/512` L1 terms.

- [ ] **Step 4: Verify GREEN**

Run the temporary test and expect all mel tests to pass.

### Task 3: Full-audio epoch media and Aim reporting

**Files:**
- Create: `src/runner/nodes/training/styletts3/testing/vocoder_training/reporting.py`
- Modify temporarily: `/tmp/test_istftnet2_training.py`

**Interfaces:**
- Produces: `EpochReporter(output_dir, aim_run, sample_rate)`, `save_validation_item(epoch, index, ground_truth, prediction, gt_mel, pred_mel)`, `track_train`, `track_validation`, and `close`.
- Consumes: Aim `Run`, NumPy/torch audio, and two mel tensors.

- [ ] **Step 1: Add a failing artifact test**

Use a recording fake Aim run and assert that one call writes `gt.wav`, `pred.wav`, and a nonempty `mel.png` under the epoch/item directory and tracks two Aim audio objects plus one image.

- [ ] **Step 2: Verify RED**

Run the temporary test and expect the reporting import to fail.

- [ ] **Step 3: Implement media export and scalar/media tracking**

Write PCM-16 WAVs with SoundFile, render paired mel panels with shared limits using the noninteractive matplotlib backend, track finite metrics, and track Aim `Audio` and `Image` values with epoch/global-step coordinates.

- [ ] **Step 4: Verify GREEN**

Run the temporary test and expect the reporting test to pass.

### Task 4: GAN optimization, full validation, and checkpoints

**Files:**
- Create: `src/runner/nodes/training/styletts3/testing/vocoder_training/trainer.py`
- Create: `src/runner/nodes/training/styletts3/testing/vocoder_training/__init__.py`
- Modify temporarily: `/tmp/test_istftnet2_training.py`

**Interfaces:**
- Produces: `TrainingConfig`, `train_batch()`, `validate_epoch()`, `save_checkpoint()`, and `train_vocoder()`.
- Consumes: models, loader, validation entries, mel modules, reporter, optimizers, and CUDA device.

- [ ] **Step 1: Add failing optimizer/checkpoint tests**

On a small real model batch, assert finite discriminator/generator/mel/adversarial/feature metrics, changed generator and discriminator weights, frozen-state restoration after the generator step, and a checkpoint containing both models, both optimizers, epoch, and global step.

- [ ] **Step 2: Verify RED**

Run the temporary test and expect missing trainer interfaces.

- [ ] **Step 3: Implement the train and validation loops**

Compute conditioning once per batch, retain one generator forward for both updates, detach fake audio for the discriminator update, freeze discriminator parameters during the generator update, use bfloat16 autocast on CUDA, aggregate metrics, validate held-out files sequentially in full, report media, and save one resumable checkpoint per epoch plus `generator_final.pth`.

- [ ] **Step 4: Verify GREEN**

Run the temporary test and expect the full suite to pass.

### Task 5: Runnable CLI, smoke execution, and real run

**Files:**
- Create: `src/runner/nodes/training/styletts3/testing/train_istftnet2_mb.py`
- Remove: `/tmp/test_istftnet2_training.py`

**Interfaces:**
- Produces: `python -m runner.nodes.training.styletts3.testing.train_istftnet2_mb ...`.
- Consumes: backend dataset UUID, `AIM_REPO`, output path, batch/worker values, and optional smoke limits.

- [ ] **Step 1: Implement strict CLI parsing and Aim initialization**

Require `--dataset-id` and `--output-dir`, expose batch/workers/epochs/validation/max-items/max-steps, require an initialized `AIM_REPO`, record hyperparameters, assert CUDA, build fused Adam optimizers, and invoke `train_vocoder`.

- [ ] **Step 2: Run static and temporary behavioral verification**

Run `py_compile`, the complete temporary suite, `git diff --check`, file/folder size checks, then remove the temporary test with `apply_patch` and repeat `py_compile`/`diff --check`.

- [ ] **Step 3: Run the real backend-connected smoke job**

Run through Nix with LJSpeech dataset `022c31b6-83be-4cd6-9835-25aa3830357b`, one epoch, one train step, a tiny train subset and two validation files. Verify the checkpoint, WAV, PNG, metrics, and Aim run.

- [ ] **Step 4: Inspect parameter counts**

Only now print generator and discriminator parameter counts and confirm no behavior was tuned to them.

- [ ] **Step 5: Launch the five-epoch job**

Run the same module with five epochs, 16 validation files, batch size 16, and optimized worker/prefetch settings. Monitor startup through the first successful batch and report the run/output/Aim locations.
