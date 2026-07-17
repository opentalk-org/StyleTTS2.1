# Beetle Training Baseline Design

## Purpose and scope

Beetle is a configurable 100M–150M parameter text-to-speech model trained in
three standalone Python stages. The initial delivery is scripts plus strict
configuration, not a Runflow node. Training logic must remain independent of
CLI and Runflow lifecycle details so a later node can supply database
selection, cancellation, progress, metrics, and artifact callbacks without
rewriting models, datasets, losses, or trainers.

The parameter target covers inference-time Beetle modules. It excludes the
prompt `TextEncoder`, frozen helper models, discriminators, and other
training-only modules. Prompt-to-style and prompt-to-voice training is outside
the three stages.

The latent-to-audio path has a separate compute ceiling, not a parameter
ceiling. It must remain below 15 GFLOPs per generated second. Parameter count is
still reported; a roughly 35M-parameter full-width decoder/generator path is
acceptable when it satisfies the compute ceiling.

## Package structure

Implementation lives mainly in `src/runner/nodes/training/beetle/`:

```text
beetle/
├── config/                 # strict YAML configuration
├── data/                   # database index, sampling, decoding, collators
├── models/                 # portable model package
│   ├── model.py            # public model composition and builders
│   ├── complexity.py       # inference profiling and limits
│   └── modules/            # model implementations and internal layers
├── losses/                 # focused loss implementations
├── training/               # reusable trainers, state, validation, callbacks
├── scripts/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── train_stage3.py
├── external/               # shallow pinned reference repositories
├── files/                  # explicitly required supporting files
└── papers/                 # primary papers, extracted text, implementation notes
```

Files stay below 300 lines and folders below 16 files. The lowercase `models/`
package follows the shallow StyleTTS2 layout: `model.py` is the public
composition entry point, while `models/modules/` owns complete network
implementations and their internal layers. Large duration and latent-flow
implementations may use lowercase subpackages under `models/modules/`.
Losses, data access, and training lifecycle code remain outside `models/`.

## Architecture

All temporal tensors use `[batch, channels, time]`. Architecture topology is
explicit Python code. Widths, depths, kernels, dilation schedules, latent
sizes, dropout, conditioning dimensions, and pretrained identifiers are
validated configuration fields; configuration does not dynamically assemble
an arbitrary network graph.

### Latent audio path

The style-free reconstruction path is:

```text
mel[T] -> AudioEncoder -> z[T/2]
z -> FeatureLinear -> F0/N[T]
z, F0, N -> Decoder -> h[T], prepared F0[T]
h, prepared F0 -> Generator -> waveform[T*300]
```

- `AudioEncoder` is a Piper/VITS-inspired dilated residual posterior encoder.
  It downsamples the hop-300 mel sequence by exactly two and returns posterior
  mean, log scale, and sampled half-rate latent `z`. Collation pads full-rate
  frame counts to an even length and retains the true mask and sample length.
- `FeatureLinear` applies its framewise projection to `z`, then deterministic
  linear interpolation by two produces full-rate F0 and normalized log-energy
  `N` tracks. Training targets remain at hop 300.
- `Decoder` is a style-free adaptation of the current
  `StyleTTSISTFTNet2MBDecoder`'s StyleTTS2 `DecoderBackbone`. It preserves the
  full-width encode/decode topology, stride-two F0/N projections, repeated
  latent/F0/N injection, and final temporal upsampling. AdaIN is replaced by
  learned, style-independent instance normalization. It returns typed
  full-rate synthesis features and the prepared F0 used by the Generator.
- `Generator` is a style-free multiband iSTFTNet2-MB waveform synthesizer with
  F0 harmonic excitation and a 300-sample output hop.

StyleTTS2 nests its generator inside `Decoder`; Beetle keeps Decoder and
Generator separate while preserving its tensor geometry and conceptual order.
Style and voice affect synthesized half-rate latents through `LatentFlowModel`,
not the audio decoder or generator. Phoneme alignment and duration supervision
stay at the hop-300 full-rate clock. Expanded phoneme conditioning is padded to
an even full-rate length and pairwise pooled before it enters the half-rate
latent flow. Waveform and F0/N targets also stay at the hop-300 full-rate clock.

### Latent-to-audio complexity gate

Complexity preflight profiles `FeatureLinear`, `Decoder`, and `Generator`
together in evaluation mode with batch size one. The canonical benchmark uses
40 half-rate latent frames and produces 80 hop-300 frames, or exactly one second
at 24 kHz. FLOPs use `torch.utils.flop_counter.FlopCounterMode`, matching the
repository convention where a multiply-accumulate counts as two operations.
The report contains total parameters, total FLOPs, generated seconds, normalized
GFLOPs per output second, and an explicit over-budget flag. Preflight fails
before training when normalized compute is at least 15 GFLOPs per second.

The unchanged current StyleTTS3 reference measures 35.223M parameters and
4.054 GFLOPs for this benchmark. Beetle may retain comparable full decoder
widths because the measured compute is below the approved ceiling. The final
Beetle implementation is measured independently; reference measurements are
context, not proof that the implementation passes.

### Text, context, duration, and latent generation

- `PhonemeEncoder` wraps a custom BERT loaded from a configured local
  directory. The same directory supplies its tokenizer. Loading is local-only;
  Beetle does not compare checkpoint vocabulary or hidden-width metadata
  against separate expected values.
- `architecture.phoneme_token_count` is the only phoneme vocabulary-size
  setting and defaults to 178. The data/tokenization contract and
  StyleTTS2-compatible aligner construction consume this value. Phoneme and
  aligner sub-configurations do not duplicate it. Strict checkpoint loading and
  normal tensor indexing remain responsible for surfacing incompatible files.
- `LatentPhonemeEncoder`, `DurationPhonemeEncoder`, and
  `ContextPhonemeEncoder` are separate residual CNN projections.
- `ContextAudioEncoder` receives audio immediately before or after the target
  audio. Context may have a different voice.
- `TextEncoder` is a configurable multilingual BERT with prompt projection
  heads. It is implemented for the future prompt stage but excluded from the
  three current trainers.
- `StyleEncoder` and `VoiceEncoder` have separate weights and consume
  `AudioEncoder` latents using attentive statistics pooling.
- `DurationPredictor` is an invertible conditional normalizing flow over log
  phoneme durations. It supports likelihood training and reverse sampling.
- `LatentFlowModel` is a temporal CNN receiving noisy latent `x_t`, independent
  per-token noise levels `t`, per-token shortcut steps `d`, masks, and projected
  conditioning tokens.
- `PhonemeAligner` is initialized from the StyleTTS2-compatible pretrained
  aligner and produces soft attention, CTC logits, and monotonic hard
  alignment.
- `F0Extractor` is the frozen StyleTTS2-compatible pitch model. `N` is
  normalized log mel-frame energy.

### Conditioning

Every condition starts with its own linear projection into a common channel
width:

- phoneme embeddings expanded through alignment;
- style vector repeated across tokens;
- voice vector repeated across tokens;
- pooled phoneme vector repeated across tokens;
- pre/post text context applied to the first/last `k` phonemes;
- pre/post audio context applied to the corresponding boundary region.

`k` is sampled uniformly from 1 through 32. Projected condition sequences are
concatenated into selected CNN layers, and combined conditioning drives
AdaLN-Zero in residual blocks. Conditions are not prematurely reduced to one
global vector.

Dropout is independent for every sample and condition source. A dropped source
becomes an exact zero tensor, so one batch can contain arbitrary mixtures of
conditioning. All probabilities are configurable. Initial configuration uses
approximately 1% phoneme-embedding dropout and 75% pre/post text and audio
context dropout; style, voice, and pooled-phoneme dropout have separate fields.
Dataset context availability and model-side dropout are different masks.

## Database-backed data pipeline

The database schema is the source contract:

- a `Dataset` contains audio files through `dataset_audio_files`;
- an `AudioFile` owns packed WAV storage, metadata, file-level `style_prompt`
  and `voice_prompt`, and a JSONB segment array;
- each segment contains `start`, `end`, `text`, `phon`, `speaker`, `voice_id`,
  confidence, metadata, and optional word-level alignment;
- `list_segment_references_page` provides cursor-paged metadata;
- `list_audio_segments_bulk` retrieves current JSONB segment payloads;
- `bulk_read_wav_segments` performs bounded, grouped packed-audio reads.

The scripts accept `dataset_id` and an optional selected audio-ID set through
typed configuration. A future node passes the same selection from a
`TrainingManifest`. JSONL is not the primary source.

`DatabaseSegmentIndex` scans references in cursor pages and retains a compact
index: audio ID, segment index and ID, time range, estimated byte cost,
`voice_id`, eligibility flags, prompt availability, and per-audio temporal
order. Full segment JSON is not retained. Batch prefetch retrieves current
segments for unique audio IDs in bulk and verifies stored segment IDs against
the index so database edits cannot silently change a resumed run.

All target, context, style-pair, and voice-pair ranges for multiple planned
batches are deduplicated and bulk-read together. Prefetch is bounded by both
planned batch count and estimated decoded bytes. Audio is downmixed, resampled,
and converted to mel features through one configured path. F0 and `N` are
computed in batches on the training device.

`voice_id` is the canonical voice label. Free-form `speaker` is diagnostic and
is not silently substituted. Style and voice prompts are optional file-level
text; the schema has no prompt-audio field. Virtual metadata-only audio is not
eligible for this baseline.

Stored alignment is word-level, not phoneme/frame alignment. The pretrained
aligner creates phoneme/frame targets during training. Complete stored segments
form the sentence sampling pool. Mid-sentence samples cut only at stored word
boundaries and are eligible when aligned words correspond to whitespace-separated
phoneme word groups. The configured sentence/mid-sentence ratio is drawn from
separate eligible pools and fails preflight if it cannot be satisfied.

Targets are 1–45 seconds. Pre/post text and audio use the excluded portion of a
mid-sentence cut or temporally adjacent segments in the same audio file;
adjacent segments may have another `voice_id`. The collator pads every input
independently and returns explicit lengths, validity masks, context-availability
masks, alignments, pair views, labels, and optional prompt tokens.

Reconstruction sampling is duration-bucketed. Voice batches contain multiple
utterances per voice for GE2E. Style batches contain distance-weighted cuts
from one recording plus unrelated negatives. Time stretch, pitch shift, and
gain affect embedding views without corrupting reconstruction or alignment
targets.

Preflight reports stage-specific eligibility counts. Stage 1 requires readable
stored audio and valid duration. Stages 2 and 3 additionally require text,
phonemes, and `voice_id`. Word alignment is required only for the mid-sentence
pool. Empty or incomplete future datasets fail before model setup with an
actionable eligibility report.

## Training stages and losses

### Stage 1: latent audio reconstruction

Train AudioEncoder, FeatureLinear, style-free Decoder, style-free Generator,
and the existing StyleTTS multi-period and multi-resolution spectrogram
discriminators. Losses are encoder KL, F0 MSE, `N` MSE, multiscale mel/STFT
reconstruction, the existing StyleTTS discriminator and generator losses, and
feature matching. Generator-side and discriminator-side updates are separate.

### Stage 2: duration, flow, alignment, style, and voice

Load Stage 1. AudioEncoder runs frozen under `no_grad` to produce target
latents. Decoder, Generator, FeatureLinear, and acoustic discriminators are not
used for training updates.

Train the custom phoneme BERT and the phoneme/context projections,
ContextAudioEncoder, DurationPredictor, LatentFlowModel plus EMA,
PhonemeAligner, StyleEncoder, VoiceEncoder, the style speaker classifier, and
the style-statistics head.
Losses are:

- duration normalizing-flow negative log likelihood over aligned log duration;
- latent conditional flow matching with independent per-token noise levels;
- shortcut EMA bootstrap consistency for configured dyadic step sizes;
- sequence-to-sequence, monotonic soft/hard, and CTC aligner losses;
- voice contrastive and GE2E losses;
- distance-weighted style contrastive and recording-grouped GE2E losses;
- gradient-reversal speaker removal from style;
- regression of F0/N mean and standard deviation from style;
- latent generation and StyleEncoder re-encoding consistency.

Prompt-to-style and prompt-to-voice losses are not part of this stage.

### Stage 3: end-to-end fine-tuning

Load both prior checkpoints and unfreeze inference-time Beetle modules. Train
both existing StyleTTS discriminator families in Stage 3. Run a posterior
reconstruction path and a text-conditioned shortcut path using aligned training
durations and differentiable one-step latent generation through Decoder and
Generator. Combine Stage 1 acoustic/adversarial losses with Stage 2 duration,
flow, alignment, embedding, and consistency losses. Every loss weight and
activation schedule is step-based configuration.

## Generative-loss research gate

Duration-flow and latent-flow losses must not be connected from memory. Before
implementation, pin shallow StyleTTS2 and Piper references in `external/`, and
save the primary VITS, Flow Matching, Diffusion Forcing, Shortcut Models,
StyleTTS2, and relevant vocoder papers plus extracted text in `papers/`.

Implementation notes must explicitly derive and cross-check:

- normalizing-flow forward/reverse transforms and change-of-variables log
  determinants;
- duration-flow likelihood and masking;
- the selected conditional probability path and analytic flow-matching target;
- independent per-token noise construction;
- shortcut dyadic step distribution, EMA bootstrap target, and stop-gradient
  boundaries;
- one-step and multi-step sampling equations.

The whole Beetle directory remains below 20 GB. It contains no datasets, model
weights, generated audio, or run outputs.

## Continuous step-based execution

There is no epoch concept. Each stage continuously cycles through eligible
sampling pools until cancellation. At the end of a deterministic permutation,
the sampler increments `cycle_index`, reshuffles, and continues. Learning-rate,
loss, logging, checkpoint, and validation schedules use optimizer steps only.

Validation runs when `optimizer_step % validation_every_steps == 0`. A fixed
validation selection is persisted for comparable results across resumes.
Outputs are saved below `validation/step_<optimizer_step>/` as WAV files plus
typed JSON metadata:

- Stage 1 saves reference audio, posterior reconstruction, predicted F0/N, and
  reconstruction metrics.
- Stage 2 saves reference audio, complete text-conditioned synthesis through
  frozen Stage 1 decoding, duration/alignment visualizations, and latent-flow
  metrics.
- Stage 3 saves reference, posterior reconstruction, end-to-end synthesis, and
  complete acoustic, discriminator, duration, alignment, flow, style, and voice
  metrics.

## Exact resume and lifecycle adapters

Every checkpoint contains all model and discriminator states, EMA, optimizers,
schedulers, AMP scalers, optimizer and accumulation microsteps, accumulated
gradients, Python/NumPy/Torch CPU/CUDA RNG states, sampler cycle/permutation and
next-batch position, data-index fingerprint, loss-schedule state, configuration
fingerprint, and fixed validation selection.

Augmentation, cutting, context selection, conditioning dropout, and generative
noise derive deterministic seeds from stage, cycle, batch, sample ID, and view
ID. Prefetched but unconsumed work is never recorded as consumed. A completed
checkpoint resumes at the exact next microbatch. Graceful cancellation writes
an atomic checkpoint before exit. Abrupt termination resumes exactly from the
latest successful atomic checkpoint; checkpoint frequency is configurable.

The core receives cancellation, progress, metrics, validation-artifact, and
checkpoint callbacks. CLI callbacks are implemented now. A future node maps
these to `context.check_cancel()`, `context.report_progress()`, shared CRUD, and
artifact publication without changing training behavior.

## Verification and failure behavior

Strict configuration rejects unknown fields and incompatible dimensions before
loading models or data. Schema drift, segment-ID drift, invalid alignment,
checkpoint incompatibility, and NaN/Inf losses fail explicitly. Samples are not
silently skipped.

Temporary verification run through `nix develop --command` covers model tensor
and mask contracts, normalizing-flow inversion and log determinants, analytic
flow/shortcut targets, mixed conditioning within batches, all loss backward
passes, optimizer ownership, parameter count, the latent-to-audio complexity
gate, all three CLI preflights, and save/resume equivalence against an
uninterrupted synthetic run. Complexity verification asserts the exact
half-rate/full-rate/output geometry and fails a deliberately oversized test
configuration. Temporary tests are removed before completion because repository
policy forbids committed tests unless explicitly requested.
