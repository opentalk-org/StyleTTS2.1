# Beetle Adversarial Stages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` inline. Do not dispatch subagents for this plan.

**Goal:** Restore discriminator training and adversarial generator losses in
Stage 1 while retaining discriminator training in Stage 3 and keeping Stage 2
discriminator-free.

**Architecture:** Restore the previously implemented Stage 1 adversarial path
through the current optimizer, checkpoint, validation, and asynchronous MLflow
reporting interfaces. Reuse the shared Stage 1 schedules in Stage 3 so one
source defines the seven audio loss weights.

**Tech Stack:** Python, PyTorch, Pydantic, YAML, MLflow.

## Global constraints

- Run Python through `nix develop --command python`.
- Keep tests temporary and remove them before finishing.
- Do not restart or mutate a currently running training process.
- Stage 1 and Stage 3 train discriminators; Stage 2 does not.

### Task 1: Restore the stage contract

**Files:**

- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/config/training.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1_setup.py`
- Modify: `src/runner/nodes/training/beetle/training/loss_schedules.py`

**Interfaces:** `Stage1Schedules.weights(step) -> Stage1LossWeights` and
`build_stage1_optimizers(...) -> OptimizerSet` with `discriminator` and
`generator` groups.

- [x] Run `/tmp/beetle_stage_adversarial_check.py`; expect failure because the
  default Stage 1 discriminator optimizer is absent.
- [x] Require Stage 1 and Stage 3 discriminator optimizer configuration while
  rejecting it in Stage 2.
- [x] Restore all seven Stage 1 loss schedules and both optimizer groups.
- [x] Re-run the temporary check after the runtime changes.

### Task 2: Restore Stage 1 adversarial execution and state

**Files:**

- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/stage3.py`
- Modify: `src/runner/nodes/training/beetle/training/execution/support.py`

**Interfaces:** `Stage1Trainer.discriminator_backward(batch)` reports raw and
weighted discriminator loss; `generator_backward(batch)` reports adversarial
and feature-matching loss; checkpoints own discriminator model and gradients.

- [x] Enable the loop discriminator pass for Stage 1 and inherited Stage 3.
- [x] Restore discriminator and generator adversarial backward paths.
- [x] Include the discriminator in Stage 1 state, gradients, and gradient norms.
- [x] Restore Stage 1 discriminator weights when initializing later stages.
- [x] Remove the duplicate Stage 3 discriminator gradient group.

### Task 3: Restore validation and documentation

**Files:**

- Modify: `src/runner/nodes/training/beetle/training/validation/stage1.py`
- Modify: `src/runner/nodes/training/beetle/main.md`
- Modify: `src/runner/nodes/training/beetle/README.md`
- Modify: `docs/superpowers/specs/2026-07-17-beetle-mlflow-validation-design.md`

- [x] Evaluate and aggregate all seven Stage 1 losses plus discriminator and
  generator totals.
- [x] Replace statements that reserve discriminators for Stage 3.
- [x] Run compile checks, the temporary contract check, focused synthetic
  backward checks, file limits, and `git diff --check`.
- [x] Remove the temporary check and commit the correction.
