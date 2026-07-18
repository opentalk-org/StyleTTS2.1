# Beetle Adversarial Crop Design

## Decision

Stages 1 and 3 train the decoder, generator, reconstruction objective, and
StyleTTS discriminators on random aligned 9,600-sample segments. The audio
encoder and every non-vocoder objective continue to use the complete padded
utterance. Stage 2 remains full-sequence and unchanged.

Cropping the input before the audio encoder would reduce memory further but
would violate the full-utterance encoder requirement. Cropping only the real
and generated waveforms before the discriminator would leave the decoder and
generator activation cost unchanged. Therefore the crop boundary is between
full-rate acoustic feature prediction and the decoder.

## Geometry and configuration

The top-level adversarial training configuration contains the explicit
`segment_samples: 9600` value. Configuration loading requires this value to be
divisible by the 300-sample audio hop and requires its 32 full-rate frames to
be divisible by the posterior encoder's two-frame downsampling rate.

Each example gets an independently selected valid start frame. Starts are
aligned to the posterior downsampling rate, so one crop selects exactly:

- 9,600 waveform samples;
- 32 full-rate mel, F0, and noise frames; and
- 16 posterior-latent frames.

An utterance shorter than one crop is rejected explicitly. Padding is never
selected as training audio.

## Stage 1 data flow

The audio encoder receives the full mel tensor and mask. Posterior KL uses the
complete posterior sequence. FeatureLinear receives the complete posterior and
produces full-utterance F0 and noise predictions. Noise supervision uses the
complete valid sequence; F0 targets and supervision use the generator crop.

For each discriminator or generator pass, a deterministic per-example crop
plan is derived from the runtime seed, stage, cycle, batch index, and pass name.
That plan selects matching mel target, latent, acoustic-feature, mask, and
real-waveform segments. The frozen F0 extractor, F0 loss, Decoder, Generator,
mel/STFT reconstruction, adversarial, and feature-matching computation use only
those segments.

## Stage 3 data flow

Posterior encoding, conditioning, duration flow, latent flow, alignment,
speaker/style objectives, and full-rate acoustic feature prediction remain
full-utterance. One crop plan per discriminator pass and one per generator pass
is shared by the posterior and conditional synthesis paths. Both paths and the
real waveform therefore cover the same time interval before their losses are
averaged.

## Resume and validation

Crop plans are pure functions of checkpointed loop coordinates and the runtime
seed. A checkpoint taken after discriminator completion recreates the same
generator crop after resume without adding mutable checkpoint fields. Existing
RNG, sampler, gradient, optimizer, and discriminator state remain unchanged.

Validation remains full-utterance so saved audio and plots retain their current
meaning. Training metrics remain batch aggregates; no crop-level or sample-level
metrics are added.

## Compilation and verification

The full training batch has fixed 808-frame inputs and a fixed 32-frame vocoder
segment. AudioEncoder, FeatureLinear, and Decoder use static compiled graphs.
Generator remains eager because TorchInductor cannot lower its complex
harmonic-phase cumulative path.

Temporary checks, removed before completion, cover crop geometry, per-example
valid bounds, latent/audio alignment, deterministic plan reproduction, full
encoder and acoustic-loss lengths, Stage 1 and Stage 3 cropped adversarial
paths, and an uninterrupted versus resumed next step. A real fresh Stage 1 run
must complete discriminator and generator work at batch size 64 and publish all
seven Stage 1 loss metrics plus optimizer and gradient metrics to MLflow.
