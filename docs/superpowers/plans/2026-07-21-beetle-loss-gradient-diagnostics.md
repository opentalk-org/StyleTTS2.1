# Beetle Loss and Gradient Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 250-step Stage 1 diagnostics that attribute gradients by loss, expose spectral reconstruction detail, and report optimizer clipping strength.

**Architecture:** Extend the acoustic loss result with structured resolution and frequency-band components. Keep autograd attribution in a focused `training/diagnostics/` package, call it from Stage 1 before the normal combined backward, and add clipping observations at optimizer completion without changing optimization behavior.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, Pydantic training runtime, unittest

## Global Constraints

- Run project Python and tests through `nix develop --command ...`.
- Diagnostics run every 250 completed optimizer steps.
- Do not alter the serialized training configuration or config fingerprint.
- Do not restart or signal the active KL-off process.
- Keep each source file under 300 lines and each folder under 16 files.
- Tests are temporary and must be removed before completion.

---

### Task 1: Structured Reconstruction Breakdown

**Files:**
- Modify: `src/runner/nodes/training/beetle/losses/acoustic.py`
- Test temporarily: `/tmp/test_beetle_reconstruction_diagnostics.py`

**Interfaces:**
- Consumes: predicted waveform, target waveform, and sample mask.
- Produces: `ReconstructionLoss` with `resolutions: tuple[ResolutionLoss, ...]` and `bands: tuple[FrequencyBandLoss, ...]` in addition to the unchanged `mel` and `total` tensors.

- [ ] **Step 1: Write failing temporary tests**

Cover the three configured resolutions, the four bands `(0,1000)`, `(1000,4000)`, `(4000,8000)`, `(8000,12000)`, finite scalar results, and `total == mel == mean(resolution losses)`.

```python
loss = MultiResolutionReconstructionLoss(sample_rate=24000)
result = loss(predicted, target, torch.ones_like(target, dtype=torch.bool))
self.assertEqual(tuple(item.resolution.n_fft for item in result.resolutions), (1024, 2048, 512))
self.assertEqual(tuple((item.band.minimum_hz, item.band.maximum_hz) for item in result.bands), ((0, 1000), (1000, 4000), (4000, 8000), (8000, 12000)))
torch.testing.assert_close(result.total, torch.stack(tuple(item.value for item in result.resolutions)).mean())
```

- [ ] **Step 2: Run the tests and confirm the missing fields fail**

Run:

```bash
nix develop --command python /tmp/test_beetle_reconstruction_diagnostics.py
```

Expected: failure because `ReconstructionLoss` has no `resolutions` or `bands`.

- [ ] **Step 3: Implement the structured breakdown**

Add frozen `FrequencyBand`, `ResolutionLoss`, and `FrequencyBandLoss` dataclasses. Preserve the existing normalized relative-L1 resolution calculation, calculate band-relative L1 using theoretical HTK mel center frequencies, and average matching bands across resolutions.

- [ ] **Step 4: Run the temporary tests**

Run the same command. Expected: all reconstruction diagnostic tests pass.

### Task 2: Loss-Attributed Gradient Diagnostics

**Files:**
- Create: `src/runner/nodes/training/beetle/training/diagnostics/__init__.py`
- Create: `src/runner/nodes/training/beetle/training/diagnostics/gradients.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Test temporarily: `/tmp/test_beetle_gradient_diagnostics.py`

**Interfaces:**
- Consumes: weighted named loss tensors and declared activation interfaces.
- Produces: stable `TrainingMetric` tuples for norms and cosines without modifying `.grad`.

- [ ] **Step 1: Write failing tests for cadence and attribution**

Verify `diagnostics_due(completed_step)` is true only for positive multiples of 250. Build scalar tensor losses with known gradients, verify weighted L2 norms and cosine values, verify a zero-weight loss reports norm zero, and verify `.grad` remains `None`.

```python
self.assertFalse(diagnostics_due(0))
self.assertTrue(diagnostics_due(250))
self.assertFalse(diagnostics_due(251))
```

- [ ] **Step 2: Run the tests and confirm imports fail**

Run:

```bash
nix develop --command python /tmp/test_beetle_gradient_diagnostics.py
```

Expected: failure because the diagnostics package does not exist.

- [ ] **Step 3: Implement focused gradient helpers**

Define `DIAGNOSTICS_EVERY_STEPS = 250`, `diagnostics_due(completed_step: int) -> bool`, an L2 norm helper that accepts `None` entries but requires at least one real gradient, a cosine helper returning value and defined observations for zero-norm comparisons, and a Stage 1 attribution function using `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` only on activation interfaces.

Emit weighted metrics for:

```text
gradient_by_loss/reconstruction/waveform
gradient_by_loss/adversarial/waveform
gradient_by_loss/feature_matching/waveform
gradient_by_loss/reconstruction/generator_input
gradient_by_loss/adversarial/generator_input
gradient_by_loss/feature_matching/generator_input
gradient_by_loss/f0/acoustic_f0
gradient_by_loss/n/acoustic_n
gradient_by_loss/kl/posterior
gradient_cosine/reconstruction_adversarial
gradient_cosine/reconstruction_feature_matching
```

- [ ] **Step 4: Integrate Stage 1 diagnostics**

At each generator microstep, use `completed_step = self._loop.optimizer_step + 1`. When due, compute attributed gradients before normal backward, enable frequency-band reductions, and append reconstruction resolution/band metrics. Keep the existing generator total and backward unchanged.

- [ ] **Step 5: Run the temporary tests**

Run the same command. Expected: all gradient diagnostic tests pass and `.grad` remains untouched.

### Task 3: Optimizer Clipping Diagnostics

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/optimizer.py`
- Test temporarily: `/tmp/test_beetle_optimizer_diagnostics.py`

**Interfaces:**
- Consumes: pre-clipping norm, configured maximum norm, and completed optimizer step.
- Produces: clip coefficient and clipped flag only on 250-step diagnostic boundaries.

- [ ] **Step 1: Write failing clipping tests**

Test coefficients `1.0` for a norm below the limit and `0.05` for norm 200 with limit 10, plus numeric flags `0.0` and `1.0`.

- [ ] **Step 2: Run the tests and confirm the helper is missing**

Run:

```bash
nix develop --command python /tmp/test_beetle_optimizer_diagnostics.py
```

Expected: failure because clipping diagnostic metrics are not implemented.

- [ ] **Step 3: Implement and connect clipping metrics**

Pass an explicit Stage 1 diagnostic flag into optimizer finishing, preserve `clip_grad_norm_`, and append these metrics only when the flag is true. Stage 2 and Stage 3 retain their prior metric set:

```text
optimizer/<name>_clip_coefficient
optimizer/<name>_was_clipped
```

- [ ] **Step 4: Run the temporary tests**

Run the same command. Expected: all clipping diagnostic tests pass.

### Task 4: Focused Verification and Cleanup

**Files:**
- Verify: all modified Beetle sources
- Delete: `/tmp/test_beetle_reconstruction_diagnostics.py`
- Delete: `/tmp/test_beetle_gradient_diagnostics.py`
- Delete: `/tmp/test_beetle_optimizer_diagnostics.py`

**Interfaces:**
- Consumes: completed implementation.
- Produces: validated source with no retained test artifacts and an untouched live process.

- [ ] **Step 1: Run all temporary tests together**

```bash
nix develop --command python -m unittest discover -s /tmp -p 'test_beetle_*diagnostics.py'
```

Expected: all tests pass.

- [ ] **Step 2: Run compile and configuration checks**

Run a real two-rank CPU Gloo DDP test that performs activation-interface attribution, normal backward, and verifies synchronized parameter gradients.

```bash
nix develop --command python -m compileall -q src/runner/nodes/training/beetle
nix develop --command python -c 'from runner.nodes.training.beetle.config import load_config; load_config("src/runner/nodes/training/beetle/runs/ljspeech-stage1/config-kl-off.yaml"); print("config_ok")'
```

Expected: compile exits zero and configuration prints `config_ok` without a fingerprint-affecting schema change.

- [ ] **Step 3: Remove temporary tests and verify repository limits**

Remove the three `/tmp` files, run `git diff --check`, verify every modified source remains below 300 lines, and verify the active PID still references `config-kl-off.yaml`.

- [ ] **Step 4: Review the diff**

Confirm the optimized total, scheduled weights, normal backward, and active process are unchanged; only detached diagnostics and reporting fields are added.
