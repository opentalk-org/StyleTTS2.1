# Beetle Stage 1 Audio Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the style-free latent audio encoder, F0/N projection, Decoder, multiband iSTFTNet2-MB Generator, current StyleTTS discriminators, and Stage 1 losses.

**Architecture:** A VITS/Piper-inspired posterior encoder produces frame latents; a style-free residual decoder transforms latents with predicted F0/N; a separate harmonic multiband generator synthesizes hop-300 waveforms. Focused losses wrap current StyleTTS GAN behavior without adding discriminator families.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, current StyleTTS3 iSTFTNet2-MB and StyleTTS discriminator references, Nix.

## Global Constraints

- Decoder and Generator are separate and receive no style input.
- Generator output length is exactly latent frames multiplied by 300 samples.
- Reuse current StyleTTS multi-period and multi-resolution spectrogram discriminators; do not add Wave-U-Net or SLM models.
- Keep fixed topology with configured dimensions and every file below 300 lines.
- Use temporary CPU tests under `/tmp`; do not start a CUDA training job.

---

### Task 1: Convolution blocks, posterior encoder, and F0/N head

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/__init__.py`
- Create: `src/runner/nodes/training/beetle/models/modules/convolution.py`
- Create: `src/runner/nodes/training/beetle/models/audio_encoder.py`
- Create: `src/runner/nodes/training/beetle/models/features.py`
- Create temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `DilatedResidualStack`, `AudioPosterior(mean, log_scale, latent)`, `AudioEncoder.forward(mel, mask)`, `FeatureLinear.forward(z, mask) -> AcousticFeatures(f0, n)`, and frozen `F0Extractor.forward(waveform, lengths)`.
- Consumes: architecture/audio config from the foundation plan.

- [ ] Write failing tests with mel `(2, 80, 17)` and unequal masks. Assert posterior tensors `(2, latent_channels, 17)`, masked positions are zero, reparameterization is deterministic with a supplied generator, and FeatureLinear returns finite F0/N tracks `(2, 17)`.
- [ ] Run `nix develop --command pytest -q /tmp/test_beetle_stage1.py`; expect import failure.
- [ ] Implement gated dilated residual blocks with weight normalization, mask application after every residual operation, posterior projection to `2 * latent_channels`, clamped log scale from config, and explicit reparameterization.
- [ ] Implement one framewise linear projection with named F0/N outputs plus a strict adapter for the configured StyleTTS2-compatible pitch checkpoint; no style conditioning or hidden fallback.
- [ ] Re-run tests and a backward pass through `latent.mean() + f0.mean() + n.mean()`; expect PASS, finite predictor gradients, and no F0Extractor gradients.
- [ ] Commit: `git commit -m 'feat: add beetle audio encoder'`.

### Task 2: Style-free latent Decoder

**Files:**
- Create: `src/runner/nodes/training/beetle/models/decoder.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `Decoder.forward(z, f0, n, mask) -> Tensor[B, generator_channels, T]`.
- Consumes: `DilatedResidualStack` and configured latent/decoder channels.

- [ ] Add failing shape/mask tests and inspect `inspect.signature(Decoder.forward)` to assert it contains no style argument.
- [ ] Assert changing only F0 or `N` changes valid output while identical masked suffixes remain zero.
- [ ] Implement bias-free F0/N projections, concatenation with `z`, residual encode/decode blocks, and a final generator-channel projection at unchanged frame rate.
- [ ] Re-run focused tests; expect output `(2, generator_channels, 17)` and finite backward gradients.
- [ ] Commit: `git commit -m 'feat: add style-free beetle decoder'`.

### Task 3: Harmonic multiband iSTFT Generator

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/source.py`
- Create: `src/runner/nodes/training/beetle/models/modules/istft.py`
- Create: `src/runner/nodes/training/beetle/models/generator.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `HarmonicSource`, `MultiBandISTFT`, `Generator.forward(features, f0, mask) -> Tensor[B, 1, T*300]`.
- Consumes: explicit iSTFTNet2-MB geometry and Stage 1 Decoder output.

- [ ] Add failing tests for intermediate temporal/frequency shapes, exact hop-300 length for odd/even frame counts, deterministic harmonic phase with supplied generator, finite waveform, and no style parameter.
- [ ] Implement F0 harmonic excitation, configured temporal upsampling, 1D multi-receptive-field blocks, 2D shuffle/frequency upsampling, complex subband spectra, subband iSTFT, and multiband synthesis.
- [ ] Assert frequency bins, subband count, and final waveform length inside forward paths so invalid configuration fails at the responsible layer.
- [ ] Run generator forward/backward CPU tests; expect PASS with gradients in source and spectral branches.
- [ ] Compare tensor geometry against the current StyleTTS3 paper-profile implementation without importing testing-only model classes at runtime.
- [ ] Commit: `git commit -m 'feat: add beetle multiband generator'`.

### Task 4: Stage 1 losses and discriminator adapter

**Files:**
- Create: `src/runner/nodes/training/beetle/models/discriminators.py`
- Create: `src/runner/nodes/training/beetle/losses/__init__.py`
- Create: `src/runner/nodes/training/beetle/losses/acoustic.py`
- Create: `src/runner/nodes/training/beetle/losses/adversarial.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `build_styletts_discriminators(config)`, `AcousticLosses`, `AdversarialLosses`, `multiscale_reconstruction_loss`, and existing-objective discriminator/generator/feature losses.
- Consumes: current StyleTTS `MultiPeriodDiscriminator`, `MultiResSpecDiscriminator`, and approved audio geometry.

- [ ] Add tests that assert the builder returns exactly those two discriminator types and no Wave-U-Net/WavLM module. Record real/fake logits and feature maps for finite separate discriminator and generator losses.
- [ ] Add acoustic tests for masked KL, voiced-frame F0 MSE, valid-frame N MSE, and configured multi-resolution mel/STFT loss; padding changes must not affect values.
- [ ] Implement thin typed adapters around current StyleTTS objectives and focused acoustic reductions normalized by valid element count.
- [ ] Run separate discriminator and generator backward tests; discriminator detach rules must prevent generator gradients on the discriminator step and permit them on the generator step.
- [ ] Commit: `git commit -m 'feat: add beetle acoustic and gan losses'`.

### Task 5: Stage 1 bundle and integrated synthetic step

**Files:**
- Create: `src/runner/nodes/training/beetle/models/__init__.py`
- Create: `src/runner/nodes/training/beetle/models/bundle.py`
- Create: `src/runner/nodes/training/beetle/losses/composition.py`
- Modify temporarily: `/tmp/test_beetle_stage1.py`

**Interfaces:**
- Produces: `Stage1Models`, `build_stage1_models(config)`, `Stage1LossInput`, `Stage1LossOutput`, `compute_stage1_losses()` and `parameter_report()`.
- Consumes: every model/loss from Tasks 1–4.

- [ ] Add a synthetic batch test for the exact chain `mel -> AudioEncoder -> FeatureLinear -> Decoder -> Generator`, then compute every Stage 1 loss and perform separate discriminator/generator optimizer steps.
- [ ] Assert all configured trainable Stage 1 modules receive finite gradients, frozen F0 extraction receives none, and output length equals input mel frames times 300.
- [ ] Implement typed bundles and named loss output fields; reject missing loss weights rather than using defaults.
- [ ] Add parameter reporting split into inference, frozen helper, and training-only totals.
- [ ] Run the full temporary Stage 1 suite; expect PASS, then remove `/tmp/test_beetle_stage1.py` with `apply_patch`.
- [ ] Run compileall, file line counts, and `git diff --check`; expect success and files below 300 lines.
- [ ] Commit: `git commit -m 'feat: complete beetle stage1 models'`.
