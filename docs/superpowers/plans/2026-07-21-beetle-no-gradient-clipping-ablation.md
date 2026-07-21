# Beetle Stage 1 No-Gradient-Clipping Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Preserve the clipped Stage 1 baseline and launch a controlled, fresh no-gradient-clipping LJSpeech run.

**Architecture:** Use a separate complete training configuration and output directory. Disable clipping through an effectively infinite threshold so the existing optimizer and gradient telemetry paths remain unchanged.

**Tech Stack:** YAML, Python 3.12, PyTorch, MLflow

## Global Constraints

- Run project Python through `nix develop --command ...` as `user`.
- Preserve `output-kl-off` and its step-10,000 checkpoint.
- Change only both optimizer `maximum_gradient_norm` values from `10.0` to `1.0e+30`.
- Launch without `--resume` into `output-kl-off-no-clip`.
- Keep validation every 4,000 steps and gradient diagnostics every 250 steps.

---

### Task 1: Create and Validate the Ablation Configuration

**Files:**
- Create: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off-no-clip.yaml`

**Interfaces:**
- Consumes: the complete parsed values of `config-kl-off.yaml`.
- Produces: a valid Stage 1 configuration with generator and discriminator maximum gradient norms set to `1.0e+30`.

- [x] **Step 1: Create the configuration**

Copy the complete baseline YAML content into the new file, changing only these two mappings:

```yaml
generator_optimizer:
  maximum_gradient_norm: 1.0e+30
discriminator_optimizer:
  maximum_gradient_norm: 1.0e+30
```

- [x] **Step 2: Validate configuration equivalence**

Run a Nix Python comparison which loads both YAML files, replaces the two no-clip thresholds with `10.0`, and asserts the resulting structures are equal. Expected output: `config_equivalent`.

- [x] **Step 3: Validate through the project configuration loader**

Run:

```bash
nix develop --command python -c 'from runner.nodes.training.beetle.config import load_config; load_config("src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off-no-clip.yaml"); print("config_ok")'
```

Expected output: `config_ok`.

### Task 2: Preserve the Baseline and Start Fresh

**Files:**
- Preserve: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-kl-off/`
- Create at runtime: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-kl-off-no-clip/`

**Interfaces:**
- Consumes: the validated no-clipping configuration and the baseline's safe checkpoint boundary.
- Produces: a new user-owned Stage 1 training process with independent artifacts and MLflow run.

- [x] **Step 1: Confirm the baseline boundary**

Load the checkpoint referenced by `output-kl-off/checkpoints/latest.json` through Nix Python and assert `loop.optimizer_step == 10000`. Expected output: `baseline_step=10000`.

- [x] **Step 2: Gracefully stop the baseline process**

Send `SIGINT`, poll for exit, and inspect the process before considering `SIGTERM`. Do not delete or rename baseline artifacts.

- [x] **Step 3: Launch the ablation from zero**

As `user`, run the Stage 1 module through Nix with the no-clipping config and `output-kl-off-no-clip`, omitting `--resume`.

- [x] **Step 4: Verify process and MLflow identity**

Confirm the process owner and command, verify a distinct MLflow run ID, and observe initial optimizer metrics beginning from the fresh run.

### Task 3: Verify No-Clipping Telemetry

**Files:**
- Verify: `src/runner/nodes/training/beetle/runs/ljspeech-stage1/output-kl-off-no-clip/`

**Interfaces:**
- Consumes: the running no-clipping experiment.
- Produces: evidence that training reached step 250 with raw group norms retained and no group scaled.

- [x] **Step 1: Wait for optimizer step 250**

Poll MLflow in intervals below 60 seconds until the run reports step 250 diagnostics or fails.

- [x] **Step 2: Check gradient metrics**

Require `gradient/audio_encoder`, `gradient/feature_linear`, `gradient/decoder`, `gradient/generator`, and `gradient/discriminators` at step 250.

- [x] **Step 3: Check no-clipping metrics**

Require every group `clip_coefficient` to equal `1.0` and every `was_clipped` metric to equal `0.0`. Confirm the training process remains alive.

- [x] **Step 4: Check repository integrity**

Run `git diff --check` and confirm the clipped baseline checkpoint remains readable at step 10,000.
