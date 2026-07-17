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
  sampling paths. Its existing per-token linear input projection receives the
  complete condition set at phoneme rate. Sampling converts the returned
  log-duration with `ceil(exp(log_duration))` and masks padding before alignment
  expansion.
- `LatentFlowModel`: temporal CNN combining flow matching, diffusion forcing,
  and shortcut training. It receives `x_t`, independent tokenwise `t`, shortcut
  step `d`, masks, AdaLN conditioning, and condition-token concatenation at
  configured layers.
- `PhonemeAligner`: StyleTTS2-compatible pretrained aligner loaded from the
  configured checkpoint-folder UUID and filename through shared asset CRUD.
- `PhonemeEncoder`: custom BERT loaded with its tokenizer from one configured
  local directory. No additional phoneme transformer family is added.
- `LatentPhonemeEncoder`, `DurationPhonemeEncoder`, and
  `ContextPhonemeEncoder`: separate residual CNN projections.
- `ContextAudioEncoder`: consumes audio immediately before or after the target;
  context may be another speaker.
- `StyleEncoder` and `VoiceEncoder`: separate encoders over AudioEncoder
  latents.
- `LanguageEmbedding`: one learned vector per entry in the explicit ordered
  language configuration. The same selected vector conditions duration and
  latent generation.
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

Duration prediction and latent flow receive the same nine condition sources:
phoneme features, pooled phoneme vectors, style, voice, pre/post text context,
pre/post audio context, and language. Duration uses phoneme-rate phoneme
features, repeats every vector across phoneme tokens, concatenates the raw
sources, and applies its existing linear input projection. Latent flow uses
alignment-expanded latent-rate phoneme features; every source starts with its
own linear projection, including language. Projected latent conditions drive
both AdaLN and explicit token concatenation at configured CNN layers. Boundary
conditions use the first or last `k` context phonemes, with `k` sampled
uniformly from 1 to 32.

Condition dropout is independent per sample and source and uses exact zero
tensors so arbitrary condition combinations can coexist in one batch. Initial
dropout is 1% for phoneme embeddings and language and 75% for pre/post context.
One source keep decision is shared by the phoneme-rate and latent-rate paths for
each sample. Boundary context availability is decided by dataset cutting before
model dropout.

## Data source

PostgreSQL metadata is read only through shared CRUD facades. Dataset selection,
audio storage kind, waveform packs, JSONB segment payloads, alignments,
phonemes, transcripts, voice labels, and optional prompts follow the exact
shared schema. The loader builds a compact index, then bulk-prefetches current
segment JSON and deduplicated waveform ranges with bounded decoded bytes.
The configured language list is ordered and defines checkpoint-stable embedding
IDs. Audio rows with a missing language or a value absent from that list are
rejected before every stage pool is built; language is included in the dataset
fingerprint. Normal batches may mix any configured language IDs.

Training cuts are 1–45 seconds and balance sentence and mid-sentence targets.
Pre/post audio or text context is cut by the dataset pipeline and may be absent.
Style and voice grouped views support the approved contrastive and GE2E losses.
Different condition combinations are mixed within normal batches.
Reconstruction examples provide the target style/voice conditioning and
style-statistics targets; independently sampled grouped views provide the
contrastive and GE2E batches, so their different batch shapes are never
conflated. Stage 2 uses the frozen Stage 1 AudioEncoder and F0 extractor for
these targets and updates the latent-flow EMA exactly once after each completed
online optimizer step.

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
   DurationPredictor, LatentFlowModel, and the pretrained PhonemeAligner against
   posterior latents. Decoder, Generator, FeatureLinear, and discriminators are
   unused.
3. End-to-end fine-tuning with differentiable latent generation. Train both
   current discriminator families and continue fine-tuning the PhonemeAligner
   in this stage.

Stage 3 runs two audio paths per batch: posterior reconstruction and one-step
text-conditioned shortcut generation from noise. Both paths use the same
style-free Decoder followed by the separate Generator. Their F0, `N`, mel,
generator-adversarial, and feature-matching losses are averaged into the
existing Stage 1 terms, while the full Stage 2 objective remains active. The
two fake paths share one discriminator backward/update, so adversarial training
is not applied through a duplicate optimizer step.

There are no epochs. Each script samples the dataset continuously and schedules
all logging, checkpoints, and loss weights by optimizer step. There is no
validation pass, validation split, validation cadence, or validation artifact
generation. Checkpoints include model, optimizer, scaler, EMA, discriminator,
accumulated gradients, sampler cursor, and RNG state so every stage resumes
without losing a step.
Mixed Stage 2 flow batches always contain both analytic base tokens and EMA
shortcut tokens when at least two valid tokens exist, keeping their separately
weighted losses active in the same optimizer step.

## Budgets

Inference-time Beetle modules target 100M–150M parameters. TextEncoder, frozen
helpers, discriminators, and other training-only modules are excluded; only
training-only modules may be excluded.

The phoneme vocabulary contract is configured once as 178 tokens and is shared
with the aligner. The custom BERT checkpoint defines its own internal width and
parameter count; training preflight reports the actual loaded total and reports
when it exceeds the approved range. Stage 1 contributes 42,382,092 inference
parameters before the custom BERT and Stage 2 inference modules are loaded.
A BERT-base-shaped synthetic checkpoint with 178 tokens produces 199,603,199
inference parameters, which exceeds the 150M ceiling by 49,603,199. Meeting the
target therefore requires a smaller custom local BERT; startup reports the
actual loaded result.

The complete latent-to-audio path has no separate parameter ceiling but must be
below 15 GFLOPs per generated second. The canonical report uses batch one,
evaluation mode, 40 latent frames to 80 hop-300 frames and 24,000 output samples,
with PyTorch `FlopCounterMode` counting one MAC as two FLOPs. Parameter count is
always reported. Training preflight reports and rejects an over-budget path.
The current Beetle latent-to-audio implementation measures 31,364,748
parameters and 3.942411 GFLOPs per generated second on that canonical profile.
