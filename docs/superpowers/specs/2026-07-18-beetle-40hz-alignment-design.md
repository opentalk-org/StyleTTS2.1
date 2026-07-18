# Beetle 40 Hz Alignment Design

## Goal

Match StyleTTS2 alignment geometry by keeping aligner attention, monotonic
alignment, and duration supervision on the aligner's native half-mel clock.
With 24 kHz audio and hop length 300, mel features use 80 Hz while alignment
uses 40 Hz.

## Data flow

The pretrained aligner receives hop-300 mel frames and reduces time by two in
its input convolution. Its raw attention therefore has one frame for every two
mel frames. Beetle will apply masking, normalization, maximum-path extraction,
and duration summation directly to this 40 Hz attention.

The duration predictor remains phoneme-rate. Its positive integer target for
each phoneme is the number of 40 Hz hard-alignment frames assigned to that
phoneme.

The latent phoneme encoder produces one feature vector per phoneme. Batch
matrix multiplication with the hard alignment copies those vectors directly
onto the 40 Hz frame timeline:

```text
phoneme features [B,C,P] × hard alignment [B,P,F40]
    → aligned features [B,C,F40]
```

Stage 2 passes these aligned features directly to latent conditioning. It no
longer interpolates attention to 80 Hz or pairwise-pools expanded tokens back
to 40 Hz. The aligned mask must exactly match the posterior latent mask.

## Unchanged geometry

- Mel extraction remains 80 Hz.
- The audio posterior remains 40 Hz.
- The latent flow remains 40 Hz.
- FeatureLinear continues to interpolate predicted F0 and N from 40 Hz to
  80 Hz.
- The decoder continues to consume 40 Hz latents and 80 Hz F0/N, then produce
  80 Hz generator features.
- The generator continues to emit 24 kHz audio with hop 300.

## Validation and reporting

Alignment losses use native 40 Hz soft and hard matrices. Validation alignment
artifacts display the native 40 Hz matrix without presentation-only temporal
upsampling. Duration metrics are interpreted in 40 Hz frame units.

## Verification

A temporary graph-level or focused runtime check will first demonstrate that
the current adapter returns 80 Hz alignment and fails the native-rate contract.
After the change, verification will confirm:

- soft and hard alignment lengths equal the reduced aligner length;
- duration sums equal the number of valid 40 Hz frames;
- direct alignment expansion produces the posterior latent length;
- the Stage 2 input path accepts the direct 40 Hz features;
- relevant project checks pass through the Nix development environment.

Temporary verification files will be removed before completion.
