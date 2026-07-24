# Beetle Linear Band Synthesis Design

## Goal

Test whether the analytic magnitude/phase-to-waveform conversion causes the
remaining blur after the proven native-frequency head. Preserve the frequency
network and PQMF, replacing only `exp`, `sin`, complex spectrum construction,
and `torch.istft`.

## Evidence

The native-frequency head reaches mean training reconstruction `0.25604` over
steps 1,950–1,999 and improves target-aligned ridges in every frequency range.
Step-2,000 validation still shows diffuse harmonics and comb texture.

The current analytic inverse constrains the eight output channels to four
log-magnitude/phase pairs:

```text
raw pair → exp(magnitude), sin(phase) → complex spectrum → iSTFT
```

Magnitude/phase oracle swaps showed strong co-adaptation, so independently
changing either parameterization is invalid. A joint learned inverse tests the
whole mapping while leaving the upstream frequency tensor and downstream PQMF
geometry intact.

## Architecture

Reshape `[B, 8, 31, T]` into four `[B, 62, T]` coefficient streams. Apply the
same bias-free `ConvTranspose1d(62, 1, kernel=60, stride=15)` to every band.
Choose padding 23 and output padding 1 so each band has exactly `15T` samples.
Stack the four band waveforms and pass them through the unchanged `PQMF`.

The shared inverse is linear and has 3,720 parameters. The in-memory prototype
produces exactly 24,000 samples for 400 coefficient frames and costs 0.024768
GFLOPs including PQMF. Estimated combined decoder/generator compute remains
about 2.6845 GFLOPs/s.

## Isolation

Retain the already proven native-frequency and band-specific heads. Do not
change data, batch size, duration, losses, optimizer, clipping, warmups,
temporal processing, harmonic input, or PQMF.

## Verification

A temporary test must prove exact output length, zero-input/zero-output,
linearity, nonzero input and inverse-weight gradients, one shared inverse
weight tensor, and unchanged PQMF filters. The complete one-second path must
remain under 3.0 GFLOPs/s and 50 million parameters.

Train from zero through step 2,000. Compare the same 50-step windows,
validation reconstruction, four ridge-recovery bands, dark-bin leakage, and
fixed mel/STFT plots against the retained native-head run.
