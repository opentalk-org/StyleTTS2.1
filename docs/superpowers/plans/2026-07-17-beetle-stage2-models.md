# Beetle Stage 2 Conditioning and Generative Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ALBERT phoneme conditioning, context/style/voice encoders, duration normalizing flow, diffusion-forcing/shortcut latent flow, pretrained aligner integration, and every approved Stage 2 loss.

**Architecture:** Each condition is projected into a token sequence and independently zero-dropped before selected CNN concatenation and AdaLN-Zero. A verified invertible flow models log durations; a separate EMA-backed temporal CNN learns conditional latent velocity with per-token noise and shortcut steps.

**Tech Stack:** Python 3.12, PyTorch, Transformers ALBERT/BERT, verified VITS/Piper/Flow-Matching/Diffusion-Forcing/Shortcut equations, current StyleTTS aligner/F0 references, Nix.

## Global Constraints

- Implement only models named in the approved design and `main.md`.
- `PhonemeEncoder` is ALBERT; do not invent another phoneme transformer.
- Apply independent per-sample condition dropout as exact zero tensors.
- Use AdaLN plus token concatenation at explicit configured CNN layers.
- Do not implement duration or latent losses until research-note audit passes.
- Prompt TextEncoder is implemented but excluded from all three current optimizers.
- Alignment and DurationPredictor supervision remain at hop 300; pairwise pooling converts expanded phoneme conditioning to the half-rate latent clock.
- Keep every file below 300 lines and use temporary CPU tests under `/tmp`.

---

### Task 1: Conditioning modules and phoneme/context encoders

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/conditioning.py`
- Create: `src/runner/nodes/training/beetle/models/modules/text.py`
- Create temporarily: `/tmp/test_beetle_stage2.py`

**Interfaces:**
- Produces: `AdaLNZero1d`, `ConditionProjector`, `ConditionBank`, `PhonemeEncoder`, three named phoneme CNN encoders, and `ContextAudioEncoder`.
- Consumes: ALBERT model/config and Task 1 Stage 1 convolution blocks.

- [ ] Write failing tests using a tiny local `AlbertConfig`: masked ALBERT output/pool shapes, distinct latent/duration/context weights, context audio/text masked pooling, and `ConditionBank` projection to common width.
- [ ] Assert independent dropout masks can create different condition combinations in one batch and every dropped token is exactly zero after biased projections.
- [ ] Implement ALBERT composition without another transformer layer, focused residual CNN projections, attentive/masked pooling, bias-safe zeroing, and AdaLN-Zero initialized to identity.
- [ ] Re-run tests and backward checks; expect PASS with no gradients through masked tokens.
- [ ] Commit: `git commit -m 'feat: add beetle conditioning encoders'`.

### Task 2: Style, voice, and future prompt encoders

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/embeddings.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/text.py`
- Modify temporarily: `/tmp/test_beetle_stage2.py`

**Interfaces:**
- Produces: `StyleEncoder`, `VoiceEncoder`, `StyleSpeakerClassifier`, `StyleStatisticsHead`, and `TextEncoder` with separate style/voice projections.
- Consumes: AudioEncoder latents, attentive statistics pooling, multilingual BERT.

- [ ] Add tests proving StyleEncoder and VoiceEncoder share architecture but not parameters, accept latent masks, and return normalized fixed-width vectors.
- [ ] Test gradient reversal: classifier parameters minimize voice CE while the style embedding receives sign-reversed gradients. Test the statistics head returns F0/N mean/std.
- [ ] Instantiate TextEncoder from a tiny local BERT config; assert distinct prompt projections and that `build_stage2_models()` excludes its parameters from trainable groups.
- [ ] Implement exact interfaces and re-run tests; expect PASS.
- [ ] Commit: `git commit -m 'feat: add beetle style and voice encoders'`.

### Task 3: Invertible transforms and DurationPredictor

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/duration/transforms.py`
- Create: `src/runner/nodes/training/beetle/models/modules/duration/model.py`
- Create: `src/runner/nodes/training/beetle/losses/duration.py`
- Modify temporarily: `/tmp/test_beetle_stage2.py`

**Interfaces:**
- Produces: verified flow transforms, `DurationPredictor.log_prob(duration, condition, mask, generator)`, `DurationPredictor.sample(condition, mask, generator)`, and `duration_flow_loss()`. Likelihood input is positive discrete duration; variational dequantization and the log transform belong inside the predictor.
- Consumes: conventions locked in `papers/duration-flow.md`; implementation must cite those sections.

- [x] Add an audit test that refuses to import duration code unless the research note contains source commit and exact log-determinant convention.
- [x] Add transform property tests: forward then reverse reconstructs valid values within tolerance, summed forward/reverse log determinants cancel, masked tokens are identity, and finite-difference Jacobians match analytic log determinants for a tiny tensor.
- [x] Add NLL tests against a manually computed base-density/change-of-variables case and verify padding does not change normalized loss.
- [x] Implement only the verified Piper/VITS transform sequence and reverse path; do not replace it with Gaussian regression.
- [x] Run property, likelihood, sampling, and backward tests; expect PASS and positive sampled durations after the documented inverse transform.
- [ ] Commit: `git commit -m 'feat: add beetle duration flow'`.

### Task 4: LatentFlowModel and merged flow objectives

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/latent_flow/model.py`
- Create: `src/runner/nodes/training/beetle/models/modules/latent_flow/integration.py`
- Create: `src/runner/nodes/training/beetle/losses/flow.py`
- Modify temporarily: `/tmp/test_beetle_stage2.py`

**Interfaces:**
- Produces: `LatentFlowModel.forward(x_t, t, d, conditions, mask)`, `FlowTrainingSample`, `sample_flow_training_case()`, `base_flow_loss()`, `shortcut_loss()`, and `integrate_latent_flow()`.
- Consumes: `ConditionBank`, AdaLN-Zero, verified `papers/latent-flow.md`, and an EMA model supplied by training runtime.

- [x] Add analytic tests for the documented conditional path: constructed `x_t` and velocity equal hand calculations for different `t` at every token; padding and independently noised tokens are handled exactly.
- [x] Add shortcut tests asserting `d=0` uses the base target, nonzero dyadic `d` uses two EMA half steps, EMA outputs are detached, and loss gradients enter only the online model.
- [x] Add conditioning tests that inspect hooks at configured layers and prove projected token concatenation plus AdaLN are both active; mixed source dropout must work per sample.
- [x] Assert full-rate hard alignment is padded to even length and pairwise pooled to exactly the AudioEncoder latent length without per-phoneme duration rounding.
- [x] Implement explicit residual CNN blocks, time/step embeddings, configured concat locations, masked velocity output, EMA bootstrap target, and one/multi-step integration from the verified notes.
- [x] Run finite loss/backward and one-vs-two-half-step synthetic consistency tests; expect PASS.
- [ ] Commit: `git commit -m 'feat: add beetle latent shortcut flow'`.

### Task 5: PhonemeAligner and alignment losses

**Files:**
- Create: `src/runner/nodes/training/beetle/models/modules/aligner.py`
- Create: `src/runner/nodes/training/beetle/losses/alignment.py`
- Modify temporarily: `/tmp/test_beetle_stage2.py`

**Interfaces:**
- Produces: `AlignerOutput(ctc_logits, s2s_logits, soft_alignment, hard_alignment, durations)` and `AlignmentLosses(s2s, mono, ctc)`.
- Consumes: StyleTTS-compatible pretrained aligner modules, phoneme/mel lengths, monotonic alignment search.

- [x] Build a tiny fake aligner test and assert soft alignment normalization, monotonic hard path, durations summing to valid mel frames, CTC blank handling, and padding-independent losses.
- [x] Implement a strict wrapper that loads the configured checkpoint, exposes typed outputs, and fails on vocabulary/checkpoint mismatch before training.
- [x] Implement sequence CE, soft-vs-hard monotonic loss, and CTC with exact masks and normalized denominators.
- [x] Run forward/backward tests; expect PASS with actionable failures for impossible length pairs.
- [ ] Commit: `git commit -m 'feat: add beetle phoneme alignment'`.

### Task 6: Style/voice metric and consistency losses

**Files:**
- Create: `src/runner/nodes/training/beetle/losses/embeddings.py`
- Modify temporarily: `/tmp/test_beetle_stage2.py`

**Interfaces:**
- Produces: contrastive, GE2E, speaker-adversarial, style-statistics, and latent re-encoding consistency losses with typed named results.
- Consumes: grouped batch labels/views and models from Task 2/4.

- [x] Add hand-calculated tests showing same-voice vectors attract, different voices repel, recording-grouped style GE2E uses recording identity, and distance weights alter only same-recording style positives.
- [x] Add tests for style speaker CE with reversal, F0/N mean/std regression, and cosine/MSE re-encoding consistency from generated latent through StyleEncoder.
- [x] Implement masked normalized reductions and explicit temperature/scale parameters; require valid group cardinalities instead of silently skipping invalid groups.
- [x] Run all embedding loss backward tests; expect finite gradients in the intended encoders/heads.
- [ ] Commit: `git commit -m 'feat: add beetle embedding losses'`.

### Task 7: Stage 2 bundle and full synthetic objective

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/model.py`
- Modify: `src/runner/nodes/training/beetle/losses/composition.py`
- Modify temporarily: `/tmp/test_beetle_stage2.py`

**Interfaces:**
- Produces: `Stage2Models`, `build_stage2_models(config, stage1)`, `Stage2LossInput`, `Stage2LossOutput`, and `compute_stage2_losses()`.
- Consumes: every Stage 2 model/loss and frozen Stage 1 AudioEncoder.

- [x] Create one mixed synthetic batch containing all conditioning combinations and grouped style/voice views. Compute every approved Stage 2 loss in one forward/backward pass.
- [x] Assert Stage 1 AudioEncoder, Decoder, Generator, FeatureLinear, discriminators, and prompt TextEncoder have no gradients; every intended Stage 2 parameter has finite gradients.
- [x] Implement typed model/loss composition with no raw dictionary outputs and no missing-weight defaults.
- [x] Run parameter reporting and tune only configuration widths/depths until inference-time Beetle modules total 100M–150M, excluding TextEncoder and training-only modules.
- [x] Run the complete temporary Stage 2 suite; expect PASS. Keep temporary verification outside the repository.
- [x] Run compileall, line counts, and `git diff --check`; expect success.
- [ ] Commit: `git commit -m 'feat: complete beetle stage2 models'`.
