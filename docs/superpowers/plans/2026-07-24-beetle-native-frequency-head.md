# Beetle Native-Frequency Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether native-resolution, band-specific joint magnitude/phase refinement reaches sub-0.30 mel reconstruction with sharper spectra.

**Architecture:** Preserve the complete decoder/generator except its final frequency upsampler. Upsample to 32 native-resolution channels, refine them twice, and use four compact joint magnitude/phase heads.

**Tech Stack:** Python, PyTorch, pytest, MLflow, Beetle training runtime.

## Global Constraints

- Combined decoder and generator must remain below 3.0 GFLOPs/s.
- Combined decoder and generator must remain below 50 million parameters.
- Keep batch size 64 and maximum duration 8 seconds.
- Do not change data, losses, optimizer, clipping, warmups, or schedules.
- Run commands through `./nix/run-venv.sh`.
- Do not commit.

---

### Task 1: Specify the Frequency Head

**Files:**
- Create temporarily: `/tmp/test_beetle_native_frequency_head.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/generator.py`

**Interfaces:**
- Produces: `NativeFrequencyResidual(channels: int)`
- Produces: `BandSpectrumHead(input_channels: int, hidden_channels: int)`
- Updates: `Generator._subband_spectrogram(temporal: Tensor) -> Tensor`

- [ ] Write a temporary test that instantiates `Generator`, calls
  `_subband_spectrogram` with `[2, 64, 25]`, and requires `[2, 8, 31, 25]`.
- [ ] In the same test, replace each head with a constant two-channel module
  and require concatenation order `[band0 pair, band1 pair, ...]`.
- [ ] Backpropagate `spectrum.square().mean()` and require a nonzero gradient
  for every band-head parameter.
- [ ] Run the test through `./nix/run-venv.sh python` and verify it fails
  because `native_refinement` and `band_heads` do not exist.
- [ ] Add the two small modules and replace the generator's final upsampler
  with `FrequencyUpsample(16, 32, 3)`.
- [ ] Refine the native tensor with two shared residual blocks and concatenate
  the four band-head outputs.
- [ ] Run the temporary test and require all assertions to pass.

### Task 2: Enforce Complexity Budgets

**Files:**
- Modify temporarily: `/tmp/test_beetle_native_frequency_head.py`

**Interfaces:**
- Consumes: the production `Decoder` and `Generator`
- Produces: an exact one-second FLOP and parameter assertion

- [ ] Add a 40-latent-frame decoder/generator forward producing exactly
  `[1, 1, 24000]`.
- [ ] Remove weight parametrizations only on the temporary profile instances,
  then measure with `torch.utils.flop_counter.FlopCounterMode`.
- [ ] Assert total FLOPs are below `3_000_000_000` and unique combined
  parameters are below `50_000_000`.
- [ ] Run the test and record exact values.

### Task 3: Verify the Production Edit

**Files:**
- Verify: `src/runner/nodes/training/beetle/models/modules/generator.py`
- Remove: `/tmp/test_beetle_native_frequency_head.py`

**Interfaces:**
- Consumes: Beetle model construction and config
- Produces: a verified trainable generator

- [ ] Run the temporary test once more after all edits.
- [ ] Run the existing Beetle test selection through Nix.
- [ ] Check `generator.py` remains below 300 lines and the modules folder
  remains at or below 16 files.
- [ ] Remove the temporary test as required by repository policy.
- [ ] Inspect `git diff` and confirm no unrelated production change was added.

### Task 4: Run the Isolated Ablation

**Files:**
- Use: `src/runner/nodes/training/beetle/config/default.yaml`
- Create locally: a unique run directory and log

**Interfaces:**
- Produces: MLflow metrics, validation artifacts, and a cancellation checkpoint

- [ ] Confirm no Beetle training process is active.
- [ ] Start a genuine step-zero run with the current config and a unique
  output directory.
- [ ] Monitor nonfinite steps and reconstruction trajectory without changing
  hyperparameters.
- [ ] Continue through step 2,000 and wait for validation artifacts at steps
  1,000 and 2,000.
- [ ] Compute exact mean reconstruction over steps 950–999 and 1,950–1,999.

### Task 5: Evaluate Sharpness

**Files:**
- Update: `beetle-posterior-overfit-all16-investigation.md`

**Interfaces:**
- Consumes: fixed validation WAVs and plots
- Produces: scalar and visual decision evidence

- [ ] Measure target-aligned ridge recovery for 0–1, 1–4, 4–8, and 8–12 kHz.
- [ ] Measure energy in target-dark time-frequency bins.
- [ ] Compare fixed mel and linear-STFT plots to run
  `7b1ce7555dd4450f8c76450088fd74bf`.
- [ ] Record mel, sharpness, leakage, compute, and parameter results.
- [ ] Keep or revert the architecture based on the complete success criteria,
  without committing.
