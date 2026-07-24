# Beetle Native-Frequency Head Design

## Goal

Reach mean training `posterior_reconstruction < 0.30` over steps 950–999
while producing visibly sharper mel and linear-frequency spectrograms. The
combined decoder and generator must remain below 3.0 GFLOPs per second of
24 kHz audio and below 50 million parameters.

## Evidence

The retained step-zero run reaches `0.3420707` over steps 950–999 but loses
target-aligned spectral ridges above 1 kHz. Its predicted/target pre-iSTFT
spectral contrast is `15.8%`, `16.6%`, `5.1%`, and `4.8%` across the four
PQMF bands.

An analysis/STFT/synthesis oracle over all 16 validation recordings
reconstructs at mean mel loss `0.000327`, proving the four-band spectrum,
iSTFT, and PQMF representation preserve the target. Replacing only magnitude
or phase is destructive because those predictions are co-adapted. The output
must therefore continue predicting magnitude and phase jointly.

Normal reconstruction gradients at the predicted spectrum are weaker in the
upper bands:

| PQMF band | Magnitude gradient RMS | Phase-logit gradient RMS |
| ---: | ---: | ---: |
| 0 | `0.0012558` | `0.0012380` |
| 1 | `0.0007765` | `0.0007545` |
| 2 | `0.0006928` | `0.0005972` |
| 3 | `0.0005839` | `0.0006311` |

The current frequency path collapses from 16 channels directly to the final
eight magnitude/phase channels while upsampling from 16 to 31 bins. It has no
nonlinear processing at the native 31-bin resolution and forces all bands
through the same final projection.

## Architecture

Keep the temporal network, harmonic source, first two frequency upsamplers,
iSTFT, and PQMF unchanged. Replace only the final frequency output head:

```text
16 channels at 16 bins
  → frequency upsample to 32 channels at 31 bins
  → two shared depthwise-separable residual 2-D blocks
  → four compact band heads
      32 → 8 → native residual refinement → 2
  → concatenate [band0 magnitude, phase, ..., band3 magnitude, phase]
```

Each native residual block uses leaky-ReLU, a 3×3 depthwise convolution,
leaky-ReLU, a 1×1 pointwise convolution, and an identity addition. Each band
head predicts its magnitude and phase together.

No configuration option is needed for this temporary architecture ablation.
The dimensions are intentionally fixed to the tested design.

## Boundaries

Do not change:

- batch size 64 or maximum audio duration 8 seconds;
- dataset or fixed validation recordings;
- reconstruction, GAN, feature-matching, KL, F0, or N losses;
- optimizer, clipping, warmups, or training schedule;
- temporal decoder/generator path, harmonic input, iSTFT, or PQMF.

The exact in-memory prototype produces 24,000 samples for one second and
profiles at `2.67255744` GFLOPs/s with `9,267,906` combined decoder/generator
parameters.

## Verification

Before training, a temporary test must prove:

- output geometry is `[batch, 8, 31, time]`;
- channel pairs preserve per-band magnitude/phase ordering;
- every band head receives a nonzero gradient;
- the one-second decoder/generator path is below both hard budgets.

Train from zero through step 2,000 and evaluate both the steps 950–999 and
1,950–1,999 windows. Success requires all of:

- mean `train/posterior_reconstruction < 0.30` in at least the final window,
  with the earlier window retained as a convergence-speed comparison;
- improved target-aligned ridge recovery in every frequency range;
- no increase in target-dark-bin energy leakage;
- sharper fixed validation mel and linear-STFT plots at step 2,000 on visual
  inspection.

If the run fails, retain the architecture only if it provides a proven
sharpness improvement worth composing with the next isolated correction.
