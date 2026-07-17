# Beetle Continuous Training Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exact-resumable, cancellation-driven Stage 1/2/3 training and three standalone CLIs ready for a future Runflow adapter, without a validation pass.

**Architecture:** One typed continuous loop owns sampling, accumulation, optimizer steps, checkpoint boundaries, and callbacks. Stage modules supply explicit model/loss/optimizer behavior; CLIs validate database eligibility before loading models and future nodes replace only callbacks and launch wiring.

**Tech Stack:** Python 3.12, PyTorch AMP/distributed primitives, Pydantic v2, shared database CRUD, soundfile/torchaudio, YAML, Nix.

## Global Constraints

- Training has no epoch fields, counters, schedulers, or termination condition; it cycles until cancellation.
- All logging, checkpointing, loss schedules, and learning-rate schedules use optimizer steps.
- Exact resume includes accumulated gradients and sampler/RNG state, not only the last completed optimizer step.
- Stage 1 and Stage 3 train both approved StyleTTS discriminator families; Stage 2 trains neither.
- Training has no validation split, validation cadence, or validation artifacts.
- Validate data/config/checkpoint compatibility before allocating model weights.
- Keep each file below 300 lines and use temporary tests under `/tmp`.

---

### Task 1: Callback boundary and serializable loop state

**Files:**
- Create: `src/runner/nodes/training/beetle/training/__init__.py`
- Create: `src/runner/nodes/training/beetle/training/callbacks.py`
- Create: `src/runner/nodes/training/beetle/training/state.py`
- Create temporarily: `/tmp/test_beetle_training.py`

**Interfaces:**
- Produces: `TrainingCallbacks`, `StandaloneCallbacks`, `CancellationRequested`, `StageKind`, `TrainingPhase`, `LoopState`, `RngState`, and gradient capture/restore helpers.
- Consumes: progress, cancellation, and artifact paths without importing Runflow.

- [ ] Write a failing test with a recording callback that requests cancellation after two reported optimizer steps and records typed progress/events.
- [ ] Add state round-trip tests for optimizer step, microstep, in-step training phase, sampler cursor, Python/NumPy/Torch CPU/CUDA RNG, and named parameter gradients.
- [ ] Implement a callback protocol with `check_cancel()`, `report_progress(event)`, and `publish_artifact(path, media_type)`; implement standalone signal handling behind the same protocol.
- [ ] Implement frozen typed state records and strict capture/restore functions that require every expected gradient and RNG field.
- [ ] Run: `nix develop --command pytest -q /tmp/test_beetle_training.py`; expect PASS.
- [ ] Commit: `git commit -m 'feat: add beetle training state and callbacks'`.

### Task 2: Atomic exact checkpoints

**Files:**
- Create: `src/runner/nodes/training/beetle/training/checkpoint.py`
- Modify temporarily: `/tmp/test_beetle_training.py`

**Interfaces:**
- Produces: `CheckpointPayload`, `CheckpointManager.save(payload)`, `CheckpointManager.latest()`, `CheckpointManager.load(path)`, and `validate_resume_fingerprints()`.
- Consumes: model, frozen-model, discriminator, EMA, optimizer, scheduler, scaler, `LoopState`, config fingerprint, data fingerprint, and loss-schedule state.

- [ ] Build a tiny accumulated-gradient run, save before its optimizer step, perturb every model/runtime state, reload, and assert the next update is bitwise equal to uninterrupted CPU execution.
- [ ] Test that config/data/stage mismatches fail with the named mismatching fingerprint and that an incomplete temporary checkpoint is never returned by `latest()`.
- [ ] Implement one versioned typed payload containing all approved mutable state, including the unconsumed sampler position.
- [ ] Save to a sibling temporary folder, fsync files and parent folder, then atomically rename; maintain a validated latest-manifest only after success.
- [ ] Re-run the checkpoint tests through Nix; expect PASS.
- [ ] Commit: `git commit -m 'feat: add exact beetle checkpoints'`.

### Task 3: Step schedules, optimizers, and continuous loop

**Files:**
- Create: `src/runner/nodes/training/beetle/training/optimizer.py`
- Create: `src/runner/nodes/training/beetle/training/loop.py`
- Modify temporarily: `/tmp/test_beetle_training.py`

**Interfaces:**
- Produces: `StepSchedule.value(optimizer_step)`, `OptimizerSet`, `StageTrainer` protocol, and `run_continuously(pipeline, trainer, callbacks, checkpoint_manager)`.
- Consumes: `BeetleBatch`, resumable sampler state, exact checkpoint components, and strict training config.

- [ ] Test warmup/decay and scheduled loss weights at exact step boundaries without a dataset-pass variable.
- [ ] Test accumulation: sampler consumption advances per fetched batch, optimizer step advances only after configured microsteps, and checkpoint restore reproduces the next batch and update.
- [ ] Test cancellation between fetch, forward, discriminator update, generator update, and checkpoint operations; every completed substep updates `TrainingPhase`, and every path writes one final atomic checkpoint.
- [ ] Implement named optimizer/scheduler/scaler groups with disjoint parameter ownership assertions and serialize every group by name.
- [ ] Implement an unbounded loop that calls `pipeline.next_batch()`, checks cancellation at safe boundaries, marks a batch consumed only after its gradients and training phase are checkpoint-recoverable, rejects non-finite named losses, and triggers work with modulo step intervals.
- [ ] Bound only the temporary test via a callback that raises `CancellationRequested`; do not add a production maximum-step setting.
- [ ] Run the loop suite through Nix; expect PASS.
- [ ] Commit: `git commit -m 'feat: add continuous beetle training loop'`.

### Task 4: Stage 1 trainer

**Files:**
- Create: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify temporarily: `/tmp/test_beetle_training.py`

**Interfaces:**
- Produces: `Stage1Trainer` implementing the `StageTrainer` protocol.
- Consumes: Stage 1 models/losses, both discriminator families, `BeetleBatch`, optimizer groups, AMP scaler, and loss schedules.

- [ ] Run one synthetic step and assert 18 mel frames produce 9 AudioEncoder frames, 18 FeatureLinear/Decoder frames, and 5400 waveform samples while KL/F0/N/reconstruction/adversarial/feature-matching losses update the intended modules; F0 targets come from the frozen extractor and N targets from normalized log mel-frame energy.
- [ ] Assert multi-period and multi-resolution spectrogram discriminator parameters both update from detached real/fake audio and generator gradients do not enter the discriminator update.
- [ ] Implement separated discriminator and generator passes, gradient accumulation, scheduled weights, clipping, scaler handling, and typed metric output.
- [ ] Save/reload mid-accumulation and assert the next Stage 1 update equals uninterrupted execution.
- [ ] Run Stage 1 runtime tests through Nix; expect PASS.
- [ ] Commit: `git commit -m 'feat: add beetle stage1 trainer'`.

### Task 5: Stage 2 trainer and EMA bootstrap

**Files:**
- Create: `src/runner/nodes/training/beetle/training/stage2.py`
- Modify temporarily: `/tmp/test_beetle_training.py`

**Interfaces:**
- Produces: `Stage2Trainer` implementing `StageTrainer` and exact EMA updates.
- Consumes: frozen Stage 1 AudioEncoder, Stage 2 models/losses, source-group labels, flow sample seeds, and optimizer groups.

- [ ] Run one mixed-conditioning batch and assert independent zero-dropout patterns coexist while all Stage 2 objectives produce finite named metrics.
- [ ] Assert Stage 1 AudioEncoder runs under `no_grad`; FeatureLinear, Decoder, Generator, both discriminators, and prompt TextEncoder are unused and unchanged.
- [ ] Assert shortcut targets use detached pre-update EMA weights, then update EMA exactly once after the online optimizer step.
- [ ] Implement stateless per-sample flow/dropout seeds derived from stage, cycle, batch, sample, and view identifiers.
- [ ] Save/reload before an optimizer/EMA update and assert online and EMA next states equal uninterrupted execution.
- [ ] Run Stage 2 runtime tests through Nix; expect PASS.
- [ ] Commit: `git commit -m 'feat: add beetle stage2 trainer'`.

### Task 6: Stage 3 joint trainer

**Files:**
- Create: `src/runner/nodes/training/beetle/training/stage3.py`
- Modify temporarily: `/tmp/test_beetle_training.py`

**Interfaces:**
- Produces: `Stage3Trainer` implementing `StageTrainer`.
- Consumes: loaded Stage 1/2 inference models, latent-flow EMA, both discriminator families, posterior and text-conditioned paths, and every Stage 1/2 loss.

- [x] Construct a joint batch and assert posterior reconstruction and text-conditioned shortcut synthesis both pass through the style-free Decoder then separate Generator.
- [x] Assert every intended inference module updates, prompt TextEncoder remains excluded, and both discriminator families update in Stage 3.
- [x] Assert the composed objective contains every Stage 1 and Stage 2 named loss with configured step weights and no duplicated adversarial update.
- [x] Implement ordered discriminator, posterior-generator, and conditional-generator work with correct detach boundaries and shared accumulation semantics.
- [x] Save/reload mid-accumulation and assert the next Stage 3 model, discriminator, optimizer, scaler, sampler, and EMA states equal uninterrupted execution.
- [x] Run Stage 3 runtime tests through Nix; expect PASS.
- [x] Commit: `git commit -m 'feat: add beetle stage3 trainer'`.

### Task 7: Remove validation lifecycle

**Files:**
- Remove: `src/runner/nodes/training/beetle/training/validation.py`
- Modify: configuration, loop state, stage trainers, exports, and documentation.

**Interfaces:**
- Removes validation configuration, validation phases, fixed validation IDs,
  validation renderer protocols, and validation artifact generation.
- Keeps logging and atomic checkpoint cadence based on optimizer steps.

- [x] Remove the validation module and its public exports.
- [x] Remove validation state from exact checkpoints and the continuous loop.
- [x] Remove validation dependencies from all three stage trainers.
- [x] Remove the validation section from strict configuration.
- [x] Update the approved specification and operator documentation.

### Task 8: Runtime assembly, standalone scripts, and operator guide

**Files:**
- Create: `src/runner/nodes/training/beetle/training/runtime.py`
- Create: `src/runner/nodes/training/beetle/scripts/__init__.py`
- Create: `src/runner/nodes/training/beetle/scripts/common.py`
- Create: `src/runner/nodes/training/beetle/scripts/train_stage1.py`
- Create: `src/runner/nodes/training/beetle/scripts/train_stage2.py`
- Create: `src/runner/nodes/training/beetle/scripts/train_stage3.py`
- Create: `src/runner/nodes/training/beetle/README.md`
- Modify temporarily: `/tmp/test_beetle_training.py`

**Interfaces:**
- Produces: `prepare_run(stage, config_path, output_path, resume_path)`, `run_stage(...)`, and three `main(argv)` entry points.
- Consumes: strict config, database preflight/index, model bundles, stage trainers, callbacks, and checkpoints.

- [ ] Test each CLI parser with required config/output arguments and optional resume; reject unknown keys and any configuration containing an epoch field.
- [ ] Test an empty database/index preflight exits with stage-specific eligibility counts before monkeypatched model constructors can run.
- [ ] Implement assembly order: load/fingerprint config, build/fingerprint compact data index, validate stage eligibility and checkpoint, then load local-only phoneme BERT resources, allocate models/optimizers, and start the continuous loop.
- [ ] Keep `prepare_run()` data/checkpoint-only so empty-data preflight completes before local phoneme BERT or any other model allocation.
- [ ] Implement signal-driven cancellation and concise step/loss/checkpoint logging through `StandaloneCallbacks`.
- [ ] Document exact Nix launch commands, database requirements, stage dependencies, resume behavior, absence of validation, parameter-count exclusions, and the future callback-only Runflow adapter seam.
- [ ] Run CLI and assembly tests through Nix; expect PASS.
- [ ] Commit: `git commit -m 'feat: add beetle training scripts'`.

### Task 9: Integrated verification and cleanup

**Files:**
- Verify: `src/runner/nodes/training/beetle/`
- Keep temporary `/tmp/test_beetle_*.py` tests outside the repository.

**Interfaces:**
- Produces: a reviewed baseline ready for direct CLI use and later node adaptation.
- Consumes: every preceding linked plan.

- [ ] Run all three synthetic trainers once uninterrupted and once with save/resume; compare next-step tensors, optimizer/scaler/EMA/discriminator state, sampler position, RNG state, and metrics.
- [ ] Run `nix develop --command python -m compileall -q src/runner/nodes/training/beetle`; expect exit 0.
- [ ] Run file/folder limit checks and the master-plan byte-count command; expect project-owned files below 300 lines, project-owned folders below 16 files, and total size below 20 GB.
- [ ] Run inference parameter reporting; expect 100M–150M excluding TextEncoder, frozen helpers, discriminators, and training-only heads.
- [ ] Keep the temporary tests in `/tmp`, run `git diff --check`, and inspect `git status --short` to confirm unrelated dirty files were untouched.
- [ ] Commit: `git commit -m 'feat: complete beetle training baseline'`.
