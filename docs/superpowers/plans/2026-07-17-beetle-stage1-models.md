# Beetle Stage 1 Audio Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the half-rate posterior path, full-width style-free StyleTTS2 DecoderBackbone, separate iSTFTNet2-MB Generator, current StyleTTS discriminators, Stage 1 losses, and measured complexity gate.

**Architecture:** Hop-300 mel frames are padded to even length and encoded to a half-rate latent. FeatureLinear upsamples F0/N by two; the style-free decoder preserves the current four-block StyleTTS2 topology and restores full rate before the separate native-hop-300 generator. Model preflight measures the complete latent-to-audio path with the repository FLOP counter.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, current StyleTTS3 iSTFTNet2-MB and discriminator references, Nix.

## Global Constraints

- Decoder and Generator are separate and receive no style input.
- The decoder consumes `L` latent frames and returns `2L` synthesis frames; the Generator returns `2L * 300` samples.
- Preserve the current DecoderBackbone encode/decode count, stride-two F0/N conditioning, repeated residual conditioning, and full default widths.
- Latent-to-audio complexity is batch one, eval mode, 40 latent frames to one second at 24 kHz, using `FlopCounterMode` where one MAC is two FLOPs.
- Report parameters but reject only normalized compute at or above 15 GFLOPs per generated second.
- Reuse only current StyleTTS multi-period and multi-resolution spectrogram discriminators.
- Keep fixed topology with configured dimensions, every file below 300 lines, and temporary CPU tests under `/tmp`.

---

### Task 1: Half-rate posterior and full-rate F0/N

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/architecture.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/data/collate.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/audio.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `AudioPosterior(mean, log_scale, latent, mask)` at half rate and `FeatureLinear.forward(latent, latent_mask, frame_mask) -> AcousticFeatures(f0, n)` at full rate.
- Consumes: even padded mel `[B, 80, 2L]`, full mask `[B, 1, 2L]`, and explicit rate `2` from strict config.

- [ ] Add a failing collator/model test with source mel lengths `18` and `17`, even padded mel `(2, 80, 18)`, posterior shape `(2, 192, 9)`, F0/N shape `(2, 18)`, zero padded suffix, and reproducible sampling from equal generators; a batch whose longest source has 17 frames must pad to 18.
- [ ] Run `nix develop --command uv run --with pytest python -m pytest -q /tmp/test_beetle_stage1.py -k 'posterior or feature'`; expect shape assertions to fail against the frame-preserving implementation.
- [ ] Make mel padding round the batch maximum up to an even frame count, change the posterior input projection to configured stride-two convolution, derive the latent mask by pairwise validity, and keep mask application after every posterior residual operation.
- [ ] Keep one framewise latent-to-F0/N linear projection, apply deterministic linear interpolation by two, and mask the full-rate outputs after interpolation.
- [ ] Re-run the focused tests and backward through posterior mean plus F0/N; expect PASS, finite trainable gradients, and no gradients in the frozen F0 extractor.
- [ ] Commit the task files with `git commit -m 'feat: add half-rate beetle posterior path'`.

### Task 2: Full-width style-free DecoderBackbone

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/architecture.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/models/modules/convolution.py`
- Replace: `src/runner/nodes/training/beetle/models/modules/decoder.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `DecoderOutput(features, f0, mask)` and `Decoder.forward(latent, f0, n, latent_mask, frame_mask) -> DecoderOutput`.
- Consumes: latent `[B, 192, L]`, full-rate F0/N `[B, 2L]`, latent mask `[B,1,L]`, and full mask `[B,1,2L]`.

- [ ] Add failing tests that require no style argument, four decode blocks, default hidden width `1024`, residual width `64`, output width `512`, stride-two F0/N projections, and exact `L -> 2L` output.
- [ ] Add behavior tests proving F0 and N independently affect valid output, padded full-rate frames remain zero, training smoothing preserves shapes, and evaluation returns the original prepared F0.
- [ ] Implement a style-free residual block matching `AdainResBlk1d`: learned affine instance normalization, leaky-ReLU, dropout, weight-normalized convolutions, normalized shortcut addition, and optional nearest/depthwise-transposed-convolution upsampling.
- [ ] Implement the reference topology: encode `latent+F0+N`, project latent residual to 64 channels, reinject latent residual/F0/N before every decode block, and upsample only in the final block.
- [ ] Re-run focused forward/backward tests; expect features `(2,512,18)`, finite gradients, and exact masks.
- [ ] Commit the task files with `git commit -m 'feat: match beetle decoder backbone'`.

### Task 3: Native iSTFTNet2-MB Generator and complexity gate

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/generator.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/convolution.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/vocoder.py`
- Create: `src/runner/nodes/training/beetle/models/complexity.py`
- Modify: `src/runner/nodes/training/beetle/config/__init__.py`
- Modify: `src/runner/nodes/training/beetle/config/training.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `Generator.forward(features, f0, mask, generator) -> Tensor[B,1,T*300]`, `ComplexityReport`, `profile_latent_audio()`, and `require_complexity_budget()`.
- Consumes: `DecoderOutput`, `FeatureLinear`, canonical 40-frame latent input, and `ComplexityConfig(minimum_inference_parameters=100000000, maximum_inference_parameters=150000000, latent_audio_max_gflops_per_second=15.0, benchmark_seconds=1.0)`.

- [ ] Add generator tests for exact current 128/64 temporal geometry, native 4/8/16/31 frequency bins, four subbands, deterministic harmonic phase, and 24,000 samples from 80 synthesis frames.
- [ ] Add a failing preflight test asserting finite positive parameter/FLOP totals, one generated second, normalized GFLOPs, and explicit rejection when the configured ceiling equals the measured result.
- [ ] Keep the current iSTFTNet2-MB temporal, MRF, shuffle, frequency-upsample, multiband iSTFT, and PQMF geometry while removing only style-dependent source normalization.
- [ ] Profile `FeatureLinear -> Decoder -> Generator` under `torch.no_grad()`, eval mode, and `FlopCounterMode(display=False)`; normalize total FLOPs by generated waveform seconds.
- [ ] Run the real default profile through Nix; expect less than `15.0` GFLOPs/s. If it exceeds, report the measured modules and stop instead of silently shrinking the approved full-width topology.
- [ ] Commit the task files with `git commit -m 'feat: add beetle audio complexity gate'`.

### Task 4: Stage 1 losses and discriminator adapter

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/discriminators.py`
- Create: `src/runner/nodes/training/beetle/losses/__init__.py`
- Create: `src/runner/nodes/training/beetle/losses/acoustic.py`
- Create: `src/runner/nodes/training/beetle/losses/adversarial.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `StyleTTSDiscriminators`, `build_styletts_discriminators()`, masked acoustic losses, LSGAN discriminator/generator losses, and feature matching.
- Consumes: only current `MultiPeriodDiscriminator` and `MultiResSpecDiscriminator` families.

- [ ] Add failing tests that assert exactly those two discriminator families and no Wave-U-Net, WavLM, or SLM module.
- [ ] Add hand-calculated masked KL, voiced F0 MSE, N MSE, multiresolution reconstruction, detach-boundary, and separate backward tests.
- [ ] Implement thin typed discriminator adapters and the current StyleTTS LSGAN/feature-matching equations; normalize acoustic reductions by valid element count.
- [ ] Run focused loss tests; expect finite discriminator gradients only on its step and generator waveform gradients only on its step.
- [ ] Commit the task files with `git commit -m 'feat: add beetle acoustic and gan losses'`.

### Task 5: Stage 1 composition and synthetic step

**Files:**
- Create: `src/runner/nodes/training/beetle/models/__init__.py`
- Create: `src/runner/nodes/training/beetle/models/model.py`
- Create: `src/runner/nodes/training/beetle/losses/composition.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `Stage1Models`, `build_stage1_models()`, typed Stage 1 loss inputs/outputs, `compute_stage1_losses()`, and categorized parameter reporting.
- Consumes: all Stage 1 models, losses, strict weights, and complexity report.

- [ ] Add one synthetic test for `mel[18] -> latent[9] -> decoder[18] -> waveform[5400]`, every Stage 1 loss, and separate discriminator/generator optimizer steps.
- [ ] Assert every intended trainable module receives finite gradients, frozen F0 extraction receives none, padding does not affect losses, and complexity preflight passes.
- [ ] Implement typed bundles and named loss outputs; missing loss weights fail rather than defaulting.
- [ ] Report inference, frozen-helper, and training-only parameter totals without applying a Stage 1 parameter ceiling.
- [ ] Run the full temporary Stage 1 suite, compileall, Ruff, line counts, folder counts, and `git diff --check`; remove `/tmp/test_beetle_stage1.py` with `apply_patch` after PASS.
- [ ] Commit the task files with `git commit -m 'feat: complete beetle stage1 models'`.
