# Beetle Single-Stage Window Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the complete Beetle model in one recoverable run while feeding fixed 0.8-second acoustic segments from double-buffered, duration-grouped prefetch windows.

**Architecture:** The current joint Stage 3 objective becomes the only trainer and is initialized directly from model resources rather than dependency checkpoints. The continuous planner groups a deterministic window of uniformly sampled examples by duration, shuffles completed batches, and the prefetcher overlaps one active window with one standby window. Checkpoints contain completed optimizer boundaries only and regenerate uncommitted prefetch work on resume.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Accelerate, Pydantic, Nix development shell.

## Global Constraints

- Work in the current checkout; do not create a branch or worktree.
- Do not use subagents.
- Do not create git commits.
- Run Python and project commands through `nix develop --command ...`.
- Keep temporary tests outside the repository and delete them after verification.
- Preserve existing loss reductions; do not introduce equal-per-recording weighting.
- Use a fixed 19,200-sample acoustic segment at 24 kHz.
- Consume duration-grouped batches in deterministic random order, never monotonic length order.

---

### Task 1: One Training Configuration and Entry Point

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/training.py`
- Modify: `src/runner/nodes/training/beetle/config/data.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Create: `src/runner/nodes/training/beetle/scripts/train.py`
- Modify: `src/runner/nodes/training/beetle/scripts/common.py`
- Delete: `src/runner/nodes/training/beetle/scripts/train_stage1.py`
- Delete: `src/runner/nodes/training/beetle/scripts/train_stage2.py`
- Delete: `src/runner/nodes/training/beetle/scripts/train_stage3.py`

**Interfaces:**
- Produces: `BeetleConfig.training: TrainingConfig` and `PrefetchConfig.window_size: int`.
- Produces: `python -m runner.nodes.training.beetle.scripts.train --config ... --output ... [--resume ...]`.

- [ ] Replace the three stage configuration sections with one `training` section containing the current joint-training batch, optimizer, schedule, and loss fields.
- [ ] Remove dependency-checkpoint CLI arguments and expose only config, output, and optional resume paths.
- [ ] Set `adversarial.segment_samples` to `19200` and derive the minimum conditional mel length from this acoustic geometry.
- [ ] Add `data.prefetch.window_size`, replacing the prepared-batch-count setting while retaining decoded-byte and cache bounds.
- [ ] Run a temporary config-loading test through Nix and verify that stage keys are rejected by strict configuration.

### Task 2: Deterministic Duration-Grouped Planner Windows

**Files:**
- Modify: `src/runner/nodes/training/beetle/data/sampling.py`
- Modify: `src/runner/nodes/training/beetle/data/records.py`
- Modify: `src/runner/nodes/training/beetle/data/index.py`

**Interfaces:**
- Produces: `ContinuousBatchPlanner.next_window(batch_count: int) -> tuple[PlannedWindowBatch, ...]`.
- Produces: a planner state containing the planning frontier and any unconsumed planned batches.
- Each planned result pairs one `PlannedBatch` with the exact state committed after consuming it.

- [ ] Write a temporary deterministic-planner test with mixed durations, two ranks, and a window of four batches.
- [ ] Verify the current planner fails the grouping and randomized-batch-order assertions.
- [ ] Draw the same global number of examples as ordinary fixed batching, sort examples by target duration, partition fixed global batches, and deterministically shuffle completed batches.
- [ ] Preserve complete voice/style auxiliary groups and assign each group set exactly once to an emitted batch.
- [ ] Include pending planned batches in planner state so a checkpoint can resume midway through a window without duplication or omission.
- [ ] Verify identical seeds reproduce identical window order, order is not monotonically sorted, all examples occur once, and rank shards are disjoint.

### Task 3: Two-Buffer Window Prefetch

**Files:**
- Modify: `src/runner/nodes/training/beetle/data/prefetch.py`
- Modify: `src/runner/nodes/training/beetle/data/pipeline.py`
- Modify: `src/runner/nodes/training/beetle/data/__init__.py`

**Interfaces:**
- Consumes: `ContinuousBatchPlanner.next_window(...)` from Task 2.
- Produces: `WindowPrefetcher`, retaining `next_batch()`, `mark_consumed()`, `state_dict()`, `load_state_dict()`, and `close()` for the training loop.

- [ ] Write a temporary fake-loader test that blocks standby preparation and records fetch/consume order.
- [ ] Verify the existing batch queue cannot express active/standby window swaps.
- [ ] Make the producer prepare one complete duration-grouped window at a time and place at most one ready standby window behind the active window.
- [ ] Keep decoded-byte reservations until each prepared batch is consumed and reject a single window whose estimate exceeds the configured byte budget.
- [ ] Commit only the state attached to a consumed batch; rebuilding from committed state must deterministically recreate all prefetched but unconsumed work.
- [ ] Verify training consumes the active window while the standby window fills, swaps only at a window boundary, and closes without a live producer thread.

### Task 4: Single Joint Trainer and Optimizers

**Files:**
- Create: `src/runner/nodes/training/beetle/training/trainer.py`
- Create: `src/runner/nodes/training/beetle/training/setup.py`
- Modify: `src/runner/nodes/training/beetle/training/loss_schedules.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_inputs.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_features.py`
- Modify: `src/runner/nodes/training/beetle/training/__init__.py`
- Delete: `src/runner/nodes/training/beetle/training/stage1.py`
- Delete: `src/runner/nodes/training/beetle/training/stage2.py`
- Delete: `src/runner/nodes/training/beetle/training/stage3.py`
- Delete: `src/runner/nodes/training/beetle/training/stage1_setup.py`
- Delete: `src/runner/nodes/training/beetle/training/stage2_setup.py`

**Interfaces:**
- Produces: `BeetleTrainer`, `build_optimizers(...)`, `build_latent_flow_ema(...)`, and trainable/frozen module enumerations.
- The trainer owns acoustic models, conditioning models, EMA, discriminator and generator optimizers, and all joint losses from optimizer step zero.

- [ ] Write a temporary construction test asserting unique optimizer ownership and intended trainability for every model family.
- [ ] Verify construction from random acoustic/conditioning model initialization without stage dependency checkpoints.
- [ ] Extract the current joint objective into `BeetleTrainer` without inheritance from a stage-specific trainer.
- [x] Use a 1,000-step GAN warmup for the 2,000-step rapid test run and retain the existing generator/discriminator update order.
- [ ] Keep pretrained F0, text reference, and alignment resources frozen where their architecture declares them fixed.
- [ ] Verify one synthetic graph microstep produces gradients for acoustic, conditioning, duration-flow, latent-flow, generator, and discriminator trainable groups.

### Task 5: Fixed 0.8-Second Acoustic Segments

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/segments.py`
- Modify: `src/runner/nodes/training/beetle/data/collate.py`
- Modify: `src/runner/nodes/training/beetle/training/trainer.py`

**Interfaces:**
- Produces: aligned tensors of 19,200 samples, 64 acoustic frames, and 32 latent frames for every acoustic/GAN pass.

- [ ] Write a temporary segment test covering 0.4-, 0.8-, and 45-second valid lengths.
- [ ] Verify a short item retains valid audio and receives only right-side zero padding; verify it cannot select an all-padding region.
- [ ] Derive the conditional collator minimum frame count from `segment_samples / hop_length`.
- [ ] Select starts from valid lengths for long items and frame zero for padded short items, retaining latent alignment.
- [ ] Verify waveform, frame, and latent slices remain aligned and exactly match fixed acoustic geometry.

### Task 6: One Runtime, Validation Path, and Completed-Step Checkpoints

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/state.py`
- Modify: `src/runner/nodes/training/beetle/training/checkpoint.py`
- Modify: `src/runner/nodes/training/beetle/training/loop.py`
- Modify: `src/runner/nodes/training/beetle/training/loop_events.py`
- Modify: `src/runner/nodes/training/beetle/training/runtime.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/stages.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/services.py`
- Modify: `src/runner/nodes/training/beetle/training/reporting/mlflow.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/stage3.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/render.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/types.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/__init__.py`
- Delete: `src/runner/nodes/training/beetle/training/validation/stage1.py`
- Delete: `src/runner/nodes/training/beetle/training/validation/conditional.py`

**Interfaces:**
- Produces: `run_training(...) -> LoopState` with no stage selector or dependency checkpoints.
- Produces: checkpoint version 6 containing completed-boundary model/optimizer/EMA/scheduler/scaler/RNG/planner/reporting state and no saved gradients.

- [ ] Write temporary checkpoint tests for completed-step save/load and mid-microstep cancellation.
- [x] Reduce `StageKind` to the single training lifecycle and remove stage transition loading, dependency payloads, and stage-specific reporting/artifact names.
- [ ] Permit checkpoint creation only with `microstep == 0` at a completed optimizer boundary; remove gradient capture and restoration.
- [ ] On cancellation or failure during a partial optimizer step, discard partial gradients and retain the last completed checkpoint rather than serializing an inferred phase.
- [ ] Make `--resume` load exactly its saved optimizer step and committed planner state, then regenerate uncommitted prefetch windows.
- [ ] Verify a resumed two-buffer run produces the same next sample keys and optimizer step as uninterrupted execution.

### Task 7: Repository Verification and Cleanup

**Files:**
- Modify: documentation or exports found stale by repository-wide searches.
- Delete: temporary tests created for Tasks 1–6.

**Interfaces:**
- Produces: a repository with no callable three-stage training entry points or stale stage configuration.

- [ ] Run `rg` for stage selectors, dependency checkpoint arguments, and deleted trainer imports; remove every stale runtime reference.
- [ ] Run Ruff/formatting through `nix develop --command` using the repository's configured commands.
- [ ] Run the existing Beetle test suite through `nix develop --command python -m pytest ...`.
- [ ] Start or attach to the shared development session with `nix develop --command runflow-dev-session` and submit the relevant real graph smoke workflow through `POST /graphs/runs`.
- [ ] Inspect the run with `nix develop --command python -m cli runs`, logs, node logs, and failure output as applicable.
- [ ] Remove all temporary test files and report any verification limitation with exact evidence.
