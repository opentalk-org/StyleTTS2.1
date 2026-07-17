# Beetle MLflow and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents and worktrees are prohibited for this project.

**Goal:** Give all three Beetle stages finite step limits, exact-resume MLflow reporting, deterministic validation metrics, and the approved stage-specific artifacts.

**Architecture:** The training loop remains the lifecycle owner. Beetle-local `reporting` and `validation` packages provide typed state and services; stage trainers provide loss and inference operations without importing StyleTTS3 implementation modules. Checkpoints bind training, reporting, validation, and MLflow run state at optimizer boundaries.

**Tech Stack:** Python 3.12, PyTorch, Pydantic, MLflow client, psutil, pynvml, matplotlib, soundfile, shared PostgreSQL/audio CRUD.

## Global Constraints

- Work in the current checkout; do not create branches, worktrees, or subagents.
- Run Python and verification commands through `nix develop --command`.
- Keep files below 300 lines and folders below 16 files; use lowercase names.
- Do not import implementation modules from another node family; copy/adapt the StyleTTS3 reporting pattern into Beetle.
- Do not add an epoch field or epoch-based behavior.
- Do not commit tests, generated plots/audio, database fixtures, caches, or run output.
- Temporary tests live under `/tmp/beetle_mlflow_validation_tests` and are removed before the final commit.
- MLflow is required and fail-fast; validation rejects missing or incomplete configured samples.

---

### Task 1: Strict finite-stage configuration and optimizer ownership

**Files:**
- Create: `src/runner/nodes/training/beetle/config/validation.py`
- Modify: `src/runner/nodes/training/beetle/config/__init__.py`
- Modify: `src/runner/nodes/training/beetle/config/training.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1_setup.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_setup.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`

**Interfaces:** `ValidationConfig(audio_file_ids: tuple[UUID, ...])`; `StageConfig.total_steps`; `StageConfig.validation_every_steps`; Stage 1 owns only the generator optimizer and four acoustic losses; Stage 3 alone owns discriminator/adversarial schedules and state. The example YAML sets `total_steps: 1000000` and `validation_every_steps: 10000` for each stage, Stage 1/2 discriminator optimizers to null, and one explicit nil UUID validation ID that intentionally fails preflight until replaced with collected data.

- [ ] Write `/tmp/beetle_mlflow_validation_tests/test_config.py` with Pydantic checks that missing step fields fail, duplicate/empty validation IDs fail, Stage 1/2 reject discriminator optimizers, Stage 3 requires one, and an `epoch_count` key fails.
- [ ] Run `nix develop --command pytest /tmp/beetle_mlflow_validation_tests/test_config.py -q`; expect failures for missing `ValidationConfig` and the old Stage 1 discriminator rule.
- [ ] Implement strict fields and validators. Refactor `Stage1Schedules` to acoustic weights only; add Stage 3-only adversarial schedules. Set `Stage1Trainer.trains_discriminator = False`, remove its adversarial forward/backward and discriminator checkpoint state, keep its unused discriminator frozen on CPU, and make `Stage3Trainer._state_modules()` append the discriminator explicitly.
- [ ] Update `restore_stage1()` so dependency checkpoints restore the four acoustic modules and frozen F0 extractor, while Stage 3 initializes and subsequently checkpoints its own discriminator.
- [ ] Run the temporary test and `nix develop --command python -m compileall -q src/runner/nodes/training/beetle`; expect PASS and exit 0.
- [ ] Commit production files with `git commit -m "refactor: reserve beetle discriminators for stage three"`.

### Task 2: Checkpointed metric, timing, and validation state

**Files:**
- Create: `src/runner/nodes/training/beetle/training/reporting/__init__.py`
- Create: `src/runner/nodes/training/beetle/training/reporting/state.py`
- Create: `src/runner/nodes/training/beetle/training/reporting/metrics.py`
- Create: `src/runner/nodes/training/beetle/training/reporting/timing.py`
- Modify: `src/runner/nodes/training/beetle/training/checkpoint.py`
- Modify: `src/runner/nodes/training/beetle/training/callbacks.py`

**Interfaces:** `MetricAccumulator.add(items: int, metrics: tuple[TrainingMetric, ...])`; `MetricAccumulator.complete() -> dict[str, float]`; immutable `TimingState`; immutable `ReportingState(mlflow_run_id, timing, accumulator, last_reported_step, last_validated_step, completion)`; `CheckpointPayload.reporting`.

- [ ] Write `test_reporting_state.py` that adds two unequal microstep metric sets, asserts name-wise averages and actual item count, round-trips state through `torch.save`, and verifies the first optimizer step contributes no timing interval.
- [ ] Run the test; expect import failure for `training.reporting`.
- [ ] Implement typed accumulator and timing state. Reject duplicate metric names within one microstep, missing names across accumulation, non-finite values, negative durations, and step regression. Implement ETA exactly as `(elapsed / measured_steps * total_steps) - elapsed` and exclude process downtime on resume.
- [ ] Bump `CHECKPOINT_VERSION`, add required reporting state to `CheckpointPayload`, validation, and every trainer payload/restore call. Extend typed progress events with completed item count and foreground timing, without raw dictionaries.
- [ ] Run the test; expect PASS. Commit with `git commit -m "feat: checkpoint beetle reporting state"`.

### Task 3: Pre-clipping gradients and step observations

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/optimizer.py`
- Modify: `src/runner/nodes/training/beetle/training/loop.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2.py`
- Modify: `src/runner/nodes/training/beetle/training/stage3.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_setup.py`

**Interfaces:** `NamedGradientModule(name: str, module: nn.Module)`; `OptimizerSet.step(step, modules) -> tuple[TrainingMetric, ...]`; `StageTrainer.gradient_modules()` returns approved stage-specific groups; loop emits one completed-step observation containing averaged losses, optimizer metrics, item count, and timings.

- [ ] Write `test_step_metrics.py` with a two-parameter module whose unclipped norm is known, two accumulation microsteps with batch sizes 3 and 2, and a clip threshold below the known norm.
- [ ] Run it; expect the current optimizer API and loop to fail the contract.
- [ ] Split optimizer stepping into unscale, module/global norm measurement, clipping, stepping, scaling, and zeroing. Prefix optimizer metrics later in the reporter, while preserving typed raw names here.
- [ ] Expose Stage 1 groups `audio_encoder`, `feature_linear`, `decoder`, `generator`; Stage 2 groups for phoneme/context/conditioning/style/voice/duration/latent flow; Stage 3 combines them and adds `discriminators`.
- [ ] Aggregate every loss across all contributing microsteps and count `len(batch.sample_keys)`. Run the test; expect exact pre-clip norms and item count 5. Commit with `git commit -m "feat: observe beetle optimizer steps"`.

### Task 4: Strict asynchronous MLflow reporter

**Files:**
- Create: `src/runner/nodes/training/beetle/training/reporting/mlflow.py`
- Create: `src/runner/nodes/training/beetle/training/reporting/system.py`
- Create: `src/runner/nodes/training/beetle/training/reporting/reporter.py`
- Modify: `src/runner/nodes/training/beetle/training/reporting/__init__.py`

**Interfaces:** `MlflowSession.start(stage, resolved_config)`, `MlflowSession.resume(run_id, stage)`, `submit(metrics, step)`, `log_artifact(path, artifact_path)`, `flush()`, `finish()`, `fail()`; `TrainingReporter.complete_step(observation) -> ReportingState`; experiment `beetle_training`; maximum 256 pending metric operations.

- [ ] Write `test_mlflow_reporter.py` with a fake `MlflowClient` and controllable pending operations. Assert one `log_batch(..., synchronous=False)` per step, bounded backpressure at 256, error propagation, exact resume run ID, prefixes, and `FINISHED`/`FAILED` termination.
- [ ] Run it; expect import failure for the MLflow session.
- [ ] Copy/adapt the asynchronous pattern into Beetle. Require `MLFLOW_TRACKING_URI`; create one stage run, log resolved config, verify resumed run tags/stage, retain pending operations, and never return a no-op tracker.
- [ ] Sample CPU, system memory, RSS, GPU utilization/memory/temperature/power and include it in the same step batch. Emit throughput, ETA, foreground percentages, pending counts, and queue utilization from checkpointed timing state.
- [ ] Run the test; expect PASS. Commit with `git commit -m "feat: report beetle training to mlflow"`.

### Task 5: Ordered full-recording validation data

**Files:**
- Create: `src/runner/nodes/training/beetle/data/validation.py`
- Modify: `src/runner/nodes/training/beetle/data/__init__.py`
- Modify: `src/runner/nodes/training/beetle/training/runtime.py`

**Interfaces:** `ValidationRecording(audio_file_id, batch, waveform)`; `ValidationLoader.load(stage, audio_file_ids, phoneme_tokenizer, text_tokenizer) -> tuple[ValidationRecording, ...]` in exact configured order.

- [ ] Write `test_validation_data.py` using fake public audio/segment CRUD adapters. Cover order preservation, duplicate/missing/virtual/unreadable records, incomplete Stage 2 metadata, complete segment concatenation, disabled augmentation/context dropout, and full-frame retention.
- [ ] Run it; expect import failure for `data.validation`.
- [ ] Resolve rows with `audio_crud.get_audio_files_bulk`, segments with `audio_crud.list_audio_segments_bulk`, and bytes/ranges with public audio CRUD. Reuse `AudioPreprocessor` and collator tensor rules, but build one deterministic full-recording batch per ID with complete segments in database order and zero context-availability masks.
- [ ] Attach the immutable validation recordings to `RunPreparation`; load before model setup so all sample failures are preflight failures.
- [ ] Run the test; expect PASS. Commit with `git commit -m "feat: load ordered beetle validation audio"`.

### Task 6: Stage validation evaluators

**Files:**
- Create: `src/runner/nodes/training/beetle/training/validation/__init__.py`
- Create: `src/runner/nodes/training/beetle/training/validation/types.py`
- Create: `src/runner/nodes/training/beetle/training/validation/stage1.py`
- Create: `src/runner/nodes/training/beetle/training/validation/conditional.py`
- Create: `src/runner/nodes/training/beetle/training/validation/runtime.py`

**Interfaces:** `ValidationResult(stage, step, samples, aggregates)`; `ValidationSampleResult(audio_file_id, losses, ground_truth, prediction, latent, f0, n, mel, alignment)`; `StageValidator.evaluate(recordings, step) -> ValidationResult`.

- [ ] Write `test_validation_runtime.py` with reduced fake modules. Assert Stage 1 uses posterior reconstruction and no discriminator; Stage 2/3 use EMA latent flow with exactly one integration step; Stage 2/3 return alignment; loss aggregation is sample-count weighted; RNG and train/eval modes are byte-for-byte restored.
- [ ] Run it; expect import failure for `training.validation`.
- [ ] Implement no-grad evaluators by extracting reusable loss/inference methods from trainers rather than duplicating equations. Stage 1 returns KL/F0/`N`/StyleTTS2 mel-STFT reconstruction; Stage 2 returns every flow/alignment/style/voice objective; Stage 3 returns their union plus posterior/conditional acoustic and adversarial objectives.
- [ ] Use dedicated generators derived from stage, step, audio ID, and view. Process bounded windows when required but concatenate full audio and weight every frame once.
- [ ] Run the test; expect PASS. Commit with `git commit -m "feat: evaluate beetle stages"`.

### Task 7: Bounded validation artifacts

**Files:**
- Create: `src/runner/nodes/training/beetle/training/validation/render.py`
- Create: `src/runner/nodes/training/beetle/training/validation/artifacts.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/runtime.py`

**Interfaces:** `ArtifactQueue.enqueue(job)`, `flush()`, `close()` with bounded worker capacity; `ValidationArtifacts.publish(result) -> None`; deterministic root `validation/<stage>/step_<step>/sample_<position>_<audio_id>/`.

- [ ] Write `test_validation_artifacts.py` using a temporary output directory and blocking fake uploader. Assert backpressure, surfaced worker errors, manifest content/order, PCM WAV names, exact stage artifact sets, and no `last_validated_step` advance before successful flush.
- [ ] Run it; expect missing renderer/queue failures.
- [ ] Render detached CPU tensors with `Figure`: latent, F0, `N`, paired mel, paired STFT magnitude, paired phase, and Stage 2/3 alignment. Write `gt.wav`; use `recon.wav` only for Stage 1 and `pred.wav` for Stages 2/3. Serialize ordered per-sample/aggregate metrics to `metrics.json`.
- [ ] Submit uploads through a bounded `ThreadPoolExecutor`, retain futures, and propagate failures on flush/close. Run the test; expect PASS. Commit with `git commit -m "feat: publish beetle validation artifacts"`.

### Task 8: Finite lifecycle, resume, and CLI composition

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/loop.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/stages.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`
- Modify: `src/runner/nodes/training/beetle/training/runtime.py`
- Modify: `src/runner/nodes/training/beetle/scripts/common.py`

**Interfaces:** fresh run creates reporter; resume uses `payload.reporting.mlflow_run_id`; optimizer boundary order is optimizer completion, due validation, one metric batch, due/final flush and checkpoint; cancellation leaves MLflow active; normal completion marks it finished.

- [ ] Write `test_lifecycle.py` with fake pipeline/trainer/reporter/validator/checkpoint manager. Cover cadence, mandatory final validation, no step beyond `total_steps`, resume during accumulation, interrupted validation rerun, final checkpoint, cancellation checkpoint, and background reporting failure checkpoint.
- [ ] Run it; expect the current infinite loop to fail.
- [ ] Replace `while True` termination with the configured total-step contract. Persist reporting snapshots in every payload, place aggregate `validation/loss/*` and ordered per-sample validation losses in the same single MLflow batch as that completed step, flush before checkpoints, set `last_validated_step` only after artifact success, and make finalization idempotent for a resumed final-step checkpoint.
- [ ] Compose reporter/validator in `run_stage`; restore modes/RNG around validation; close pipeline and worker queues in deterministic order. Log concise console progress at `runtime.log_every_steps` without gating MLflow.
- [ ] Run the test; expect PASS. Commit with `git commit -m "feat: complete finite beetle stage runs"`.

### Task 9: Documentation and end-to-end verification

**Files:**
- Modify: `src/runner/nodes/training/beetle/main.md`
- Modify: `src/runner/nodes/training/beetle/README.md`

**Interfaces:** documentation describes finite stages, Stage 3-only discriminators, MLflow namespaces/lifecycle, ordered validation IDs, artifacts, and exact-resume behavior.

- [ ] Update both documents and remove statements claiming unbounded execution, Stage 1 adversarial training, or no validation.
- [ ] Run `rg -n "no validation|continuously cycles|Stage 1.*discriminator" src/runner/nodes/training/beetle/{README.md,main.md}`; expect no stale contract text.
- [ ] Run all temporary tests: `nix develop --command pytest /tmp/beetle_mlflow_validation_tests -q`; expect PASS.
- [ ] Through shared CRUD, create temporary packed audio records and a reduced local config, then run each public stage CLI with a temporary MLflow experiment. Expect exact finite stopping, required validation directories, checkpointed run IDs, and resumed runs using the same IDs. Remove the records and generated files afterward.
- [ ] Run `nix develop --command python -m compileall -q src/runner/nodes/training/beetle` and `git diff --check`; expect exit 0. Remove `/tmp/beetle_mlflow_validation_tests`.
- [ ] Commit documentation and cleanup with `git commit -m "docs: describe beetle stage reporting"`, then verify `git status --short` contains no task-created files.
