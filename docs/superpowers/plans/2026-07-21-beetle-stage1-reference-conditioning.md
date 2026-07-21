# Beetle Stage 1 Reference Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Beetle waveform synthesis with 45x mel reconstruction, log-L2 mel energy, ground-truth Stage 1 F0/N, observable posterior variance, and unclipped FeatureLinear gradients.

**Architecture:** Keep FeatureLinear predictions and their supervised losses, but pass separately computed ground-truth acoustic features into the Stage 1 decoder/generator. Centralize log-L2 mel energy in a focused acoustic module, extend existing 250-step diagnostics, and make gradient clipping policy explicit per named module.

**Tech Stack:** Python 3.12, PyTorch, Pydantic/YAML configuration, pytest through the Nix development shell.

## Global Constraints

- Do not change harmonic-source injection, decoder smoothing, KL weighting, or add custom phase/periodicity losses.
- Stage 1 waveform synthesis uses ground-truth F0 and log-L2 mel energy; FeatureLinear remains supervised.
- Stages 2 and 3 retain their predicted-conditioning routes.
- Temporary tests must be deleted before handoff.
- Run every Python command through `nix develop --command python ...`.

---

### Task 1: Log-L2 mel energy and 45x reconstruction

**Files:**
- Create: `src/runner/nodes/training/beetle/models/acoustic.py`
- Modify: `src/runner/nodes/training/beetle/models/model.py`
- Modify: `src/runner/nodes/training/beetle/models/__init__.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_features.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Test temporarily: `src/runner/nodes/training/beetle/_tmp_test_reference_conditioning.py`

**Interfaces:**
- Produces: `log_mel_l2_energy(mel: Tensor, frame_mask: Tensor) -> Tensor`.
- Consumers: `Stage1Models.n_target` and Stage 2 acoustic statistics.

- [ ] **Step 1: Write failing energy and configuration tests**

Test exact equivalence to `torch.log(torch.exp(mel).norm(dim=1))`, masking, and parsed reconstruction value 45.

- [ ] **Step 2: Verify the temporary tests fail**

Run: `nix develop --command python -m pytest src/runner/nodes/training/beetle/_tmp_test_reference_conditioning.py -v`

Expected: FAIL because `log_mel_l2_energy` does not exist and reconstruction is 5.

- [ ] **Step 3: Implement the focused energy helper and update consumers**

Move energy ownership out of `model.py`; compute the log L2 norm of linear-magnitude mel values and apply the frame mask. Change the shared Beetle reconstruction setting to 45 so every waveform reconstruction objective uses the requested weight.

- [ ] **Step 4: Verify the energy/configuration tests pass**

Run the same pytest command and expect PASS.

### Task 2: Ground-truth Stage 1 decoder conditioning

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/model.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1.py`
- Modify: `src/runner/nodes/training/beetle/training/validation/stage1.py`
- Test temporarily: `src/runner/nodes/training/beetle/_tmp_test_reference_conditioning.py`

**Interfaces:**
- Consumes: `AcousticFeatures` containing ground-truth F0/N.
- Produces: explicit target-conditioned full-recording and window reconstruction methods whose returned `Stage1Synthesis.acoustic` remains the FeatureLinear prediction.

- [ ] **Step 1: Add failing tests for conditioning separation**

Use small recording modules to assert that decoder inputs equal supplied targets, while `Stage1Synthesis.acoustic` contains FeatureLinear outputs and the F0/N losses remain differentiable with respect to FeatureLinear.

- [ ] **Step 2: Verify the tests fail because Stage 1 uses predictions**

Run the focused temporary pytest file and expect the decoder-input assertion to fail.

- [ ] **Step 3: Add explicit target-conditioned synthesis paths**

Keep the existing predicted path for Stage 3. Stage 1 generator and discriminator windows compute targets from `target_mel` and pass them to the target-conditioned window method. Stage 1 validation computes targets before synthesis and passes them to the target-conditioned full-recording method.

- [ ] **Step 4: Verify conditioning and gradient tests pass**

Run the focused temporary pytest file and expect PASS.

### Task 3: Posterior diagnostics and FeatureLinear clipping policy

**Files:**
- Modify: `src/runner/nodes/training/beetle/training/diagnostics/gradients.py`
- Modify: `src/runner/nodes/training/beetle/training/diagnostics/clipping.py`
- Modify: `src/runner/nodes/training/beetle/training/stage1_setup.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_setup.py`
- Test temporarily: `src/runner/nodes/training/beetle/_tmp_test_reference_conditioning.py`

**Interfaces:**
- Produces: explicit `GradientClipping` policy on every `NamedGradientGroup`.
- Produces at steps divisible by 250: posterior log-scale mean/min/max and mean noise scale.

- [ ] **Step 1: Add failing diagnostics and clipping tests**

Assert padded posterior positions do not affect metrics, metrics appear only on diagnostic steps, observe-only gradients remain byte-for-byte unchanged, and clipped gradients retain norm at most the configured cap.

- [ ] **Step 2: Verify the new tests fail**

Run the focused temporary pytest file and expect missing metrics/policy failures.

- [ ] **Step 3: Implement posterior metrics and explicit clipping policy**

Add the four posterior metrics to the existing diagnostic branch. Mark FeatureLinear observe-only in Stage 1 and mark every other gradient group clipped explicitly, including Stage 2 groups.

- [ ] **Step 4: Verify the focused tests pass**

Run the focused temporary pytest file and expect PASS.

### Task 4: Integrated verification and cleanup

**Files:**
- Delete: `src/runner/nodes/training/beetle/_tmp_test_reference_conditioning.py`
- Verify all modified production files.

- [ ] **Step 1: Run the complete temporary test suite once more**

Run: `nix develop --command python -m pytest src/runner/nodes/training/beetle/_tmp_test_reference_conditioning.py -v`

Expected: all tests PASS.

- [ ] **Step 2: Run compilation and configuration checks**

Run: `nix develop --command python -m compileall -q src/runner/nodes/training/beetle`

Run: `nix develop --command python -c 'from runner.nodes.training.beetle.config import load_config; c=load_config(); print(c.stage1.losses.reconstruction.value)'`

Expected: compilation exits zero and configuration prints `45.0`.

- [ ] **Step 3: Exercise the training node through a real smoke graph**

Use the shared development stack and a temporary graph/config with one Stage 1 optimizer step. Submit through `POST /graphs/runs`, then inspect the run and node logs using `nix develop --command python -m cli ...`. Do not start a second stack.

- [ ] **Step 4: Remove temporary tests and graph artifacts**

Delete temporary files with `apply_patch`; confirm they are absent with `git status --short`.

- [ ] **Step 5: Review the final diff**

Run: `git diff --check`

Run: `git diff --stat`

Expected: no whitespace errors, no generated artifacts, and no files above repository size limits.
