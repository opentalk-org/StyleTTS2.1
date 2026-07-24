# Beetle Linear Band Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace only Beetle's analytic per-band iSTFT mapping with a shared learned linear inverse while preserving PQMF.

**Architecture:** Reshape each band's 62 predicted coefficients into a shared bias-free overlapping transposed convolution, reconstruct four subband waveforms, then use the existing PQMF synthesis.

**Tech Stack:** Python, PyTorch, MLflow, Beetle training runtime.

## Global Constraints

- Keep the proven native-frequency head.
- Keep PQMF byte-for-byte unchanged.
- Keep batch size 64 and maximum duration 8 seconds.
- Combined decoder/generator must remain below 3.0 GFLOPs/s and 50M parameters.
- Do not change data, losses, optimizer, clipping, warmups, or schedules.
- Run commands through `./nix/run-venv.sh`.
- Do not commit.

---

### Task 1: Test and Implement Linear Band Synthesis

**Files:**
- Create temporarily: `/tmp/test_beetle_linear_synthesis.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/vocoder.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/generator.py`

- [ ] Write a temporary test requiring
  `MultiBandLinearSynthesis(4, 60, 15)`.
- [ ] Require `[2, 8, 31, 17]` coefficients to produce
  `[2, 1, 17 * 15 * 4]`.
- [ ] Require zero coefficients to produce an exactly zero waveform.
- [ ] Require additivity and homogeneity with tolerances suitable for float32.
- [ ] Require nonzero coefficient and inverse-weight gradients.
- [ ] Require one shared `62 × 1 × 60` bias-free inverse weight and identical
  PQMF synthesis buffers to `PQMF(4)`.
- [ ] Run the test and verify it fails because the class is absent.
- [ ] Replace `MultiBandISTFT` with the minimal learned implementation.
- [ ] Update `Generator` to construct the learned implementation.
- [ ] Run the temporary test and require all assertions to pass.

### Task 2: Verify Complexity and Repository Constraints

**Files:**
- Extend temporarily: `/tmp/test_beetle_linear_synthesis.py`

- [ ] Profile 40 decoder latent frames through one second of output.
- [ ] Require output `[1, 1, 24000]`, compute below 3 GFLOPs, and combined
  parameters below 50 million.
- [ ] Run syntax, diff, file-length, and folder-count checks.
- [ ] Remove the temporary test.

### Task 3: Train and Evaluate

**Files:**
- Create locally: unique run directory and log
- Update: `beetle-posterior-overfit-all16-investigation.md`

- [ ] Start from zero with the unchanged config.
- [ ] Train through step 2,000 and wait for both validations.
- [ ] SIGTERM only after step-2,000 artifacts finish, then verify checkpoint.
- [ ] Compare steps 950–999 and 1,950–1,999 against the native-head run.
- [ ] Compare validation mel, ridge recovery, leakage, and fixed plots.
- [ ] Keep or revert only the learned inverse based on complete evidence.
