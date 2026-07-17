# Beetle Training Baseline

Beetle is implemented first as three standalone Python training scripts with
strict configuration. The model, data, loss, and trainer APIs must remain
independent of CLI lifecycle details so a future Runflow node can provide
callbacks without rewriting training behavior.

## Models

- `AudioEncoder`: Piper/VITS-inspired posterior encoder. Hop-300 mel input is
  padded to even length and encoded to a half-rate latent sequence.
- `FeatureLinear`: one framewise latent projection for F0 and normalized
  log-energy `N`, followed by deterministic two-times interpolation.
- `Decoder`: full-width, style-free adaptation of the current StyleTTS2
  `DecoderBackbone`. It keeps stride-two F0/N conditioning, repeated latent and
  feature injection, four encode/decode stages, and final two-times temporal
  upsampling. AdaIN becomes learned style-independent instance normalization.
- `Generator`: separate style-free iSTFTNet2-MB generator using the current
  native-hop-300 temporal, harmonic-source, 1D/2D, multiband iSTFT, and PQMF
  geometry.
- `DurationPredictor`: Piper/VITS conditional normalizing flow with variational
  dequantization of positive integer durations, exact likelihood, and reverse
  sampling paths. Sampling converts the returned log-duration with
  `ceil(exp(log_duration))` and masks padding before alignment expansion.
- `LatentFlowModel`: temporal CNN combining flow matching, diffusion forcing,
  and shortcut training. It receives `x_t`, independent tokenwise `t`, shortcut
  step `d`, masks, AdaLN conditioning, and condition-token concatenation at
  configured layers.
- `PhonemeAligner`: StyleTTS2-compatible pretrained aligner.
- `PhonemeEncoder`: ALBERT. No additional phoneme transformer family is added.
- `LatentPhonemeEncoder`, `DurationPhonemeEncoder`, and
  `ContextPhonemeEncoder`: separate residual CNN projections.
- `ContextAudioEncoder`: consumes audio immediately before or after the target;
  context may be another speaker.
- `StyleEncoder` and `VoiceEncoder`: separate encoders over AudioEncoder
  latents.
- `TextEncoder`: multilingual BERT for future style/voice prompts. It is not
  trained in the current three stages and is excluded from the parameter target.
- `F0Extractor`: frozen StyleTTS2-compatible pitch model.

## Data flow

```text
mel[T] -> AudioEncoder -> z[T/2]
z -> FeatureLinear -> F0/N[T]
z + F0/N -> Decoder -> h[T] + prepared F0[T]
h + prepared F0 -> Generator -> waveform[T*300]
```

Alignment and duration supervision remain at the full hop-300 clock. Expanded
phoneme conditioning is padded to even length and pairwise pooled before the
half-rate LatentFlowModel. This avoids per-phoneme rounding drift.

Phoneme embeddings, pooled phoneme vectors, style, voice, and pre/post text or
audio context each start with their own linear projection. Style, voice, and
pooled vectors are repeated across tokens. Conditions drive both AdaLN and
explicit token concatenation at configured CNN layers. Boundary conditions are
applied to the first or last `k` phonemes, with `k` sampled uniformly from 1 to
32.

Condition dropout is independent per sample and source and uses exact zero
tensors so arbitrary condition combinations can coexist in one batch. Initial
dropout is 1% for phoneme embeddings and 75% for pre/post context. Boundary
context availability is decided by dataset cutting before model dropout.

## Data source

PostgreSQL metadata is read only through shared CRUD facades. Dataset selection,
audio storage kind, waveform packs, JSONB segment payloads, alignments,
phonemes, transcripts, voice labels, and optional prompts follow the exact
shared schema. The loader builds a compact index, then bulk-prefetches current
segment JSON and deduplicated waveform ranges with bounded decoded bytes.

Training cuts are 1–45 seconds and balance sentence and mid-sentence targets.
Pre/post audio or text context is cut by the dataset pipeline and may be absent.
Style and voice grouped views support the approved contrastive and GE2E losses.
Different condition combinations are mixed within normal batches.

## Losses

- encoder KL;
- F0 MSE on voiced valid frames and `N` MSE on valid frames;
- the StyleTTS2 three-resolution normalized log-mel spectral-convergence loss;
- current StyleTTS multi-period and multi-resolution spectrogram discriminator
  LSGAN losses plus feature matching;
- exact duration-flow likelihood;
- verified base flow-matching and shortcut objectives;
- StyleTTS2 aligner sequence-to-sequence, monotonic, and CTC losses;
- voice/style contrastive and GE2E losses;
- style speaker-adversarial, F0/N statistics, and latent re-encoding losses.

There is no Wave-U-Net, WavLM/SLM discriminator, or invented model family.

## Training stages

1. Train AudioEncoder, FeatureLinear, Decoder, Generator, and both current
   StyleTTS discriminator families.
2. Freeze Stage 1 and train the phoneme/context encoders, style/voice encoders,
   DurationPredictor, LatentFlowModel, and aligner objectives against posterior
   latents. Decoder, Generator, FeatureLinear, and discriminators are unused.
3. End-to-end fine-tuning with differentiable latent generation. Train both
   current discriminator families in this stage.

There are no epochs. Each script samples the dataset continuously and schedules
all logging, validation, checkpoints, and loss weights by optimizer step.
Validation saves stage-appropriate audio samples and typed metadata. Checkpoints
include model, optimizer, scaler, EMA, discriminator, accumulated gradients,
sampler cursor, and RNG state so every stage resumes without losing a step.

## Budgets

Inference-time Beetle modules target 100M–150M parameters. TextEncoder, frozen
helpers, discriminators, and other training-only modules are excluded; only
training-only modules may be excluded.

The complete latent-to-audio path has no separate parameter ceiling but must be
below 15 GFLOPs per generated second. The canonical report uses batch one,
evaluation mode, 40 latent frames to 80 hop-300 frames and 24,000 output samples,
with PyTorch `FlopCounterMode` counting one MAC as two FLOPs. Parameter count is
always reported. Training preflight reports and rejects an over-budget path.
