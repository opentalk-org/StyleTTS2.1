# Beetle Posterior-Only Waveform Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove conditional endpoint decoding and waveform losses from training while retaining full conditional audio synthesis during validation.

**Architecture:** Training synthesizes one posterior waveform per example and applies all acoustic and GAN losses only to that waveform. The conditional model remains in the shared generator optimizer but stops at its duration, alignment, embedding, flow-matching, shortcut, statistics, and latent consistency losses. Validation retains its independent one-step EMA endpoint synthesis for audible `full` reports.

**Tech Stack:** Python 3.12, PyTorch, Pydantic/YAML configuration, pytest, Nix development shell.

## Global Constraints

- Work in `/workspace/styletts_studio_v2`; do not create a worktree or branch.
- Do not stage or commit files and do not use subagents.
- Do not restart the active training process.
- Run Python and pytest only through `nix develop --command python -m ...`.
- Keep source files below 300 lines.
- Temporary tests must be removed after verification.
- Reconstruction weight must be exactly `45.0`.
- Validation must retain complete `step_x/full` and `step_x/audio` reports.

---

### Task 1: Establish the posterior-only training contract

**Files:**
- Create temporarily: `src/runner/nodes/training/beetle/_temporary_test_posterior_only_training.py`
- Inspect: `src/runner/nodes/training/beetle/training/trainer.py`
- Inspect: `src/runner/nodes/training/beetle/training/joint_synthesis.py`
- Inspect: `src/runner/nodes/training/beetle/config/default.yaml`

**Interfaces:**
- Consumes: current source files and resolved default YAML.
- Produces: a failing regression test describing the intended source-level training boundary.

- [ ] **Step 1: Write the failing temporary regression test**

```python
from pathlib import Path

import yaml


ROOT = Path(__file__).parent


def test_training_has_no_conditional_waveform_path() -> None:
    trainer = (ROOT / "training/trainer.py").read_text()
    assert "conditional_reconstruction" not in trainer
    assert "build_synthesis" not in trainer
    assert "generated_waveforms" not in trainer
    assert "synthesize_training_pair" not in trainer


def test_training_synthesis_has_no_latent_flow_integration() -> None:
    synthesis = (ROOT / "training/acoustic_synthesis.py").read_text()
    assert "integrate_latent_flow" not in synthesis
    assert "ConditionalSynthesis" not in synthesis
    assert "def synthesize_training_posterior(" in synthesis


def test_reconstruction_weight_preserves_posterior_coefficient() -> None:
    config = yaml.safe_load((ROOT / "config/default.yaml").read_text())
    assert config["training"]["losses"]["reconstruction"]["value"] == 45.0
```

- [ ] **Step 2: Run the temporary test and verify RED**

Run:

```bash
nix develop --command python -m pytest \
  src/runner/nodes/training/beetle/_temporary_test_posterior_only_training.py -q
```

Expected: failures because the current trainer contains `conditional_reconstruction` and `build_synthesis`, `acoustic_synthesis.py` does not exist, and reconstruction weight is `90.0`.

---

### Task 2: Replace joint training synthesis with posterior synthesis

**Files:**
- Create: `src/runner/nodes/training/beetle/training/acoustic_synthesis.py`
- Delete: `src/runner/nodes/training/beetle/training/joint_synthesis.py`
- Modify: `src/runner/nodes/training/beetle/training/conditional_features.py`
- Modify: `src/runner/nodes/training/beetle/training/conditional_inputs.py`
- Modify: `src/runner/nodes/training/beetle/training/setup.py`

**Interfaces:**
- Consumes: `AcousticModels`, `AcousticFeatures`, `AcousticSynthesis`, and `AlignedSegments`.
- Produces: `synthesize_training_posterior(acoustic_models, mel, frame_mask, segment, target, predicted_ratio, latent_generator, source_generator) -> AcousticSynthesis`.
- Removes: `ConditionalSynthesisInput` and `ConditionalInputBuilder.build_synthesis(...)`.

- [ ] **Step 1: Create the posterior-only synthesis helper**

Implement `synthesize_training_posterior` with this data flow:

```python
segment_frame_mask = segment.frames(frame_mask)
encoder_mel = segment.context_frames(mel, acoustic_models.encoder_context_frames)
encoder_mask = segment.context_frames(frame_mask, acoustic_models.encoder_context_frames)
posterior_window = acoustic_models.audio_encoder(encoder_mel, encoder_mask, latent_generator)
posterior = _slice_posterior(posterior_window, posterior_start, posterior_end)
acoustic = acoustic_models.feature_linear(
    posterior.latent,
    posterior.mask,
    segment_frame_mask,
)
segment_target = AcousticFeatures(segment.frames(target.f0), segment.frames(target.n))
decoder_acoustic = segment_target.blend(acoustic, predicted_ratio)
decoded = acoustic_models.decoder(
    posterior.latent,
    decoder_acoustic.f0,
    decoder_acoustic.n,
    posterior.mask,
    segment_frame_mask,
)
waveform = acoustic_models.generator(
    decoded.features,
    decoded.f0,
    decoded.mask,
    source_generator,
)
sample_mask = segment_frame_mask.repeat_interleave(acoustic_models.output_hop, dim=-1)
return AcousticSynthesis(posterior, acoustic, decoded, waveform, sample_mask)
```

Preserve the current posterior context-window calculation and `_slice_posterior` helper exactly. Do not import `ConditionalModels`, `ConditionalSynthesis`, `ConditionalSynthesisInput`, or `integrate_latent_flow`.

- [ ] **Step 2: Remove obsolete conditional synthesis inputs**

Delete `ConditionalSynthesisInput` from `conditional_features.py`. Remove its import and the entire `build_synthesis` method from `conditional_inputs.py`. Remove `build_synthesis` from the `ConditionalInputBuilder` protocol in `setup.py`.

- [ ] **Step 3: Remove the inaccurate joint helper**

Delete `joint_synthesis.py` after all training imports are redirected to `acoustic_synthesis.py`. Validation must continue using `validation/conditional.py::one_step_ema_latent` and its local `_synthesize` implementation.

- [ ] **Step 4: Run the temporary test to measure progress**

Run the Task 1 pytest command. Expected: the synthesis-boundary test passes; trainer and reconstruction-weight tests still fail.

---

### Task 3: Route training acoustic and GAN losses through posterior only

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/trainer.py`
- Modify: `src/runner/nodes/training/beetle/training/setup.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`

**Interfaces:**
- Consumes: `synthesize_training_posterior(...) -> AcousticSynthesis` and existing `ConditionalInputBuilder.build(...) -> ConditionalLossInput`.
- Produces: one posterior fake waveform for each real waveform and posterior-only acoustic metrics.

- [ ] **Step 1: Simplify the discriminator update**

In `discriminator_backward`, obtain targets directly and synthesize only posterior audio:

```python
target = self.acoustic.acoustic_targets(mel, frame_mask)
posterior = self._synthesize_posterior(
    mel, frame_mask, segment, target, predicted_ratio, "discriminator"
)
loss = discriminator_step_loss(
    self.acoustic.discriminators,
    real,
    posterior.waveform,
)
```

Keep synthesis under `torch.no_grad()` for the discriminator update. Do not call `input_builder`, `conditional`, or latent flow in this method.

- [ ] **Step 2: Make generator acoustic losses posterior-only**

Keep `inputs = self.input_builder.build(...)` for conditional latent losses and use `inputs.acoustic_target` for posterior synthesis. Calculate:

```python
f0 = masked_f0_smooth_l1(
    posterior.acoustic.f0,
    f0_target,
    segment_frame_mask,
)
n = masked_n_smooth_l1(
    posterior.acoustic.n,
    n_target,
    segment_frame_mask,
)
posterior_reconstruction = self.acoustic.reconstruction_loss(
    posterior.waveform,
    real,
    posterior.sample_mask,
).total
adversarial_view = generator_step_loss(
    self.acoustic.discriminators,
    real,
    posterior.waveform,
)
```

Build one scalar total from posterior acoustic losses, the one adversarial view, and `compute_conditional_losses(...)`. Call the generator optimizer's normal `backward(total / accumulation_steps)` once. Remove the waveform-gradient list, two-view loop, manual `torch.autograd.grad`, and manual `torch.autograd.backward` call.

- [ ] **Step 3: Update imports, helper name, and metrics**

Import `discriminator_step_loss`, `masked_f0_smooth_l1`, `masked_n_smooth_l1`, and `synthesize_training_posterior`. Remove imports for grouped discriminator loss, `ConditionalSynthesis`, `ConditionalSynthesisInput`, and `mean_acoustic_loss`.

Rename `_synthesize_pair` to `_synthesize_posterior` with return type `AcousticSynthesis`. Keep `posterior_reconstruction`; remove `conditional_reconstruction`.

- [ ] **Step 4: Remove the two-view acoustic loss helper**

Delete `mean_acoustic_loss` and its acoustic-loss imports from `training/setup.py`; the trainer now calls the masked loss functions directly.

- [ ] **Step 5: Set reconstruction weight to 45**

Change exactly:

```yaml
reconstruction: {value: 45.0, start_step: 0, warmup_steps: 0}
```

- [ ] **Step 6: Run the temporary test and verify GREEN**

Run the Task 1 pytest command. Expected: `3 passed`.

---

### Task 4: Make validation losses mirror training while retaining full audio

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/validation/training.py`
- Inspect unchanged behavior: `src/runner/nodes/training/beetle/training/validation/conditional.py`
- Inspect unchanged artifact routing: `src/runner/nodes/training/beetle/training/validation/artifacts.py`

**Interfaces:**
- Consumes: `ConditionalValidationSample.artifacts` as the complete audible `full` report and posterior `AcousticSynthesis` as the `audio` report.
- Produces: posterior-only aggregate acoustic validation losses plus unchanged conditional latent metrics and both artifact trees.

- [ ] **Step 1: Add a failing validation-boundary assertion**

Append to the temporary test:

```python
def test_validation_scores_posterior_but_keeps_full_synthesis() -> None:
    training = (ROOT / "training/validation/training.py").read_text()
    conditional = (ROOT / "training/validation/conditional.py").read_text()
    assert "conditional_waveform" not in training
    assert "conditional_adversarial" not in training
    assert "full = conditional.artifacts" in training
    assert "one_step_ema_latent(" in conditional
    assert "synthesis = self._synthesize(" in conditional
```

- [ ] **Step 2: Run the new assertion and verify RED**

Run the Task 1 pytest command. Expected: failure because aggregate validation still scores conditional waveform and adversarial outputs.

- [ ] **Step 3: Convert aggregate acoustic validation to posterior only**

In `_combined_sample`, retain `full = conditional.artifacts` and posterior synthesis. Remove tensors used only for conditional acoustic scoring. Compute F0, N, reconstruction, discriminator, adversarial, and feature matching from posterior values only:

```python
f0 = masked_f0_smooth_l1(posterior_f0, target_f0, frame_mask)
n = masked_n_smooth_l1(posterior_n, target_n, frame_mask)
reconstruction = self.acoustic.reconstruction_loss(
    posterior_waveform,
    real_waveform,
    sample_mask,
).total
discriminator = discriminator_step_loss(
    self.acoustic.discriminators,
    real_waveform,
    posterior_waveform,
)
adversarial_view = generator_step_loss(
    self.acoustic.discriminators,
    real_waveform,
    posterior_waveform,
)
adversarial = adversarial_view.adversarial
feature_matching = adversarial_view.feature_matching
```

Keep conditional latent metrics and `conditional_total` in `generator_total`. Do not remove or alter the full artifact set.

- [ ] **Step 4: Run the temporary tests and verify GREEN**

Run the Task 1 pytest command. Expected: `4 passed`.

---

### Task 5: Verify the complete change and remove temporary tests

**Files:**
- Delete: `src/runner/nodes/training/beetle/_temporary_test_posterior_only_training.py`
- Verify all modified Beetle files.

**Interfaces:**
- Consumes: the completed posterior-only training implementation.
- Produces: verified source tree with no persistent test artifact.

- [ ] **Step 1: Search for obsolete training symbols**

Run:

```bash
rg -n "ConditionalSynthesisInput|build_synthesis|synthesize_training_pair|conditional_reconstruction|generated_waveforms" \
  src/runner/nodes/training/beetle
```

Expected: no training-path matches. `ConditionalSynthesis` remains valid in validation.

- [ ] **Step 2: Verify validation retains full endpoint synthesis**

Run:

```bash
rg -n "one_step_ema_latent|integrate_latent_flow|full = conditional\.artifacts" \
  src/runner/nodes/training/beetle/training/validation
```

Expected: matches in `validation/conditional.py` and `validation/training.py`.

- [ ] **Step 3: Compile Beetle through Nix**

Run:

```bash
nix develop --command python -m compileall -q \
  src/runner/nodes/training/beetle
```

Expected: exit status `0` with no syntax errors.

- [ ] **Step 4: Remove the temporary test**

Delete `_temporary_test_posterior_only_training.py` with `apply_patch`.

- [ ] **Step 5: Confirm no Git staging changes and no restart**

Run `git status --short` only for inspection. Confirm the implementation remains unstaged relative to the pre-existing index and confirm the active training PID was not replaced.
