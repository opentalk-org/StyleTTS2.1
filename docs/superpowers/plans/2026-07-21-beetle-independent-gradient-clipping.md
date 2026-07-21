# Beetle Independent Gradient Clipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clip every Beetle optimizer by complete named module groups in Stages 1, 2, and 3 so one module cannot scale unrelated module gradients.

**Architecture:** Move named gradient groups into `ScheduledOptimizer` ownership and validate exact parameter coverage in `OptimizerSet`. The step path records aggregate and group pre-clip norms, independently clips each group, performs the unchanged AdamW step, and emits group plus optimizer overview diagnostics every 250 steps.

**Tech Stack:** Python 3.12, PyTorch, MLflow training metrics, unittest

## Global Constraints

- Run every project Python command through `nix develop --command ...`.
- Do not alter optimizer parameter ownership, AdamW state, loss schedules, serialized configuration, or checkpoint schema.
- Apply independent clipping to Beetle Stages 1, 2, and 3.
- Keep source files below 300 lines and folders below 16 files.
- Use temporary tests and remove them before completion.
- Keep the active Stage 1 process running until implementation verification finishes.

---

### Task 1: Optimizer-Owned Named Gradient Groups

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/optimizer.py`
- Test temporarily: `/tmp/test_beetle_group_clipping.py`

**Interfaces:**
- Consumes: `ScheduledOptimizer(..., gradient_groups: tuple[NamedGradientGroup, ...])`.
- Produces: exact ownership validation and independent group clipping from `OptimizerSet.step(optimizer_step, diagnostics=False)`.

- [ ] **Step 1: Write failing independent-clipping tests**

Create two linear modules in one AdamW optimizer with gradient norms `100` and `1`. Configure each as a separate `NamedGradientGroup`, step with maximum norm `10`, and assert the first gradient is scaled to `10` while the second remains `1`. Assert the aggregate pre-clip norm remains approximately `100.005`.

- [ ] **Step 2: Write failing ownership tests**

Assert construction fails when an optimizer parameter is missing from all groups, appears in two groups, or a group contains a trainable parameter owned by another optimizer.

- [ ] **Step 3: Run the temporary tests and confirm failure**

Run:

```bash
nix develop --command python /tmp/test_beetle_group_clipping.py
```

Expected: failure because `ScheduledOptimizer` does not accept named gradient groups and still clips optimizer-wide.

- [ ] **Step 4: Implement exact ownership and independent clipping**

Add `gradient_groups` to `ScheduledOptimizer`. In `OptimizerSet.__init__`, compare parameter identities and require every optimizer-owned parameter to appear in exactly one owned group, including parameters temporarily frozen during Stage 3 construction. Ignore frozen parameters absent from every optimizer. In `OptimizerSet.step`, record aggregate and group norms before clipping, call `clip_grad_norm_` once per group, then call the unchanged scaler/optimizer step.

Keep these metrics:

```text
optimizer/<optimizer>_gradient_norm
gradient/<group>
```

On diagnostic steps additionally emit:

```text
gradient/<group>_clip_coefficient
gradient/<group>_was_clipped
optimizer/<optimizer>_clip_coefficient
optimizer/<optimizer>_was_clipped
```

The optimizer coefficient is the minimum owned group coefficient and its flag is one when any group clips.

- [ ] **Step 5: Run the temporary tests**

Run the same command. Expected: all independent clipping, metric, and ownership tests pass.

### Task 2: Complete Cross-Stage Group Configuration

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/stage1_setup.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_setup.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2.py`
- Modify: `src/runner/nodes/training/beetle/training/stage3.py`
- Test temporarily: `/tmp/test_beetle_stage_clipping_groups.py`

**Interfaces:**
- Consumes: optimizer-owned `NamedGradientGroup` tuples from Task 1.
- Produces: exact Stage 1, 2, and 3 optimizer coverage plus 250-step diagnostics in every stage.

- [ ] **Step 1: Write failing stage coverage tests**

Build lightweight modules matching each stage grouping helper. Assert Stage 1 exposes four generator groups and one discriminator group. Assert Stage 2 group modules cover every entry returned by `trainable_stage2_modules` exactly once, including aligner and auxiliary heads. Assert Stage 3 generator groups combine Stage 1 and Stage 2 without overlap.

- [ ] **Step 2: Run the temporary tests and confirm failure**

Run:

```bash
nix develop --command python /tmp/test_beetle_stage_clipping_groups.py
```

Expected: failure because current builders do not attach groups and Stage 2 telemetry omits aligner and auxiliary heads.

- [ ] **Step 3: Attach complete groups to each optimizer**

Stage 1 generator groups:

```text
audio_encoder, feature_linear, decoder, generator
```

Stage 1 discriminator group:

```text
discriminators
```

Stage 2 groups:

```text
phoneme_encoders, context_encoders, conditioning, style_encoder,
voice_encoder, duration_predictor, latent_flow, aligner,
style_auxiliaries, voice_auxiliaries
```

Stage 3 combines the Stage 1 generator groups with all Stage 2 groups and retains the discriminator group.

- [ ] **Step 4: Make trainers use optimizer-owned groups**

Remove external `gradient_groups` arguments and duplicate trainer group methods. Call `self.optimizers.step(...)` directly. Pass `diagnostics_due(optimizer_step + 1)` from Stage 1, Stage 2, and Stage 3 so clip telemetry occurs every 250 steps in every stage.

- [ ] **Step 5: Run both temporary test files**

```bash
nix develop --command python /tmp/test_beetle_group_clipping.py
nix develop --command python /tmp/test_beetle_stage_clipping_groups.py
```

Expected: all tests pass.

### Task 3: Verification, Review, and Restart

**Files:**
- Verify: all modified Beetle training sources
- Delete: `/tmp/test_beetle_group_clipping.py`
- Delete: `/tmp/test_beetle_stage_clipping_groups.py`

**Interfaces:**
- Consumes: completed cross-stage clipping implementation.
- Produces: reviewed code and a fresh Stage 1 process with observed group clipping telemetry.

- [ ] **Step 1: Run focused verification**

```bash
nix develop --command python /tmp/test_beetle_group_clipping.py
nix develop --command python /tmp/test_beetle_stage_clipping_groups.py
nix develop --command python -m compileall -q src/runner/nodes/training/beetle
nix develop --command python -c 'from runner.nodes.training.beetle.config import load_config; load_config("src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml"); print("config_ok")'
git diff --check
```

Expected: tests pass, compilation exits zero, and configuration prints `config_ok`.

- [ ] **Step 2: Review parameter and checkpoint invariants**

Confirm AdamW parameter lists, state names, schedules, loss weights, and checkpoint targets are unchanged. Confirm every optimizer parameter is covered exactly once in all stages.

- [ ] **Step 3: Remove temporary tests and commit**

Delete both `/tmp` tests, stage only the intended optimizer/stage files, and commit the implementation.

- [ ] **Step 4: Restart Stage 1 from zero**

Gracefully stop the current KL-off process, remove only `output-kl-off/checkpoints/*`, and launch the same config without `--resume` as user through `nix develop --command ...`.

- [ ] **Step 5: Verify runtime behavior**

Confirm the new run starts at step 0. At step 250, verify log and MLflow contain group clip coefficients and that a clipped `feature_linear` group no longer changes the decoder or generator group coefficient.
