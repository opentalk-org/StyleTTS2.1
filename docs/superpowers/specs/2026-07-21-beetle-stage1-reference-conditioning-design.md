# Beetle Stage 1 reference-conditioning design

## Goal

Make Beetle Stage 1 follow the validated StyleTTS2/StyleTTS3 reconstruction
behavior where it matters, while keeping the later conditional stages and the
FeatureLinear prediction task intact.

## Scope

1. Set Beetle's shared waveform reconstruction weight to 45.
2. Replace Beetle's standardized mean-log-mel energy with the exact StyleTTS2
   `log_norm` quantity expressed for Beetle's raw log-mel representation.
3. Report posterior log-scale statistics every 250 completed Stage 1 steps.
4. Stop clipping FeatureLinear independently while preserving its gradient
   diagnostics and all other module clipping.
5. Condition Stage 1 waveform reconstruction on ground-truth F0 and energy,
   while continuing to train FeatureLinear with its supervised F0 and energy
   losses.

Source injection, decoder smoothing, KL weighting, Stages 2/3 behavior, and new
phase or periodicity losses are outside this change.

## Acoustic energy

StyleTTS2 receives normalized mel values and computes:

```text
log(norm(exp(normalized_mel * 4 - 4), dim=mel_channels))
```

Beetle stores raw log-magnitude mel values, so the equivalent target is:

```text
log(norm(exp(log_mel), dim=mel_channels))
```

The frame mask is applied to the result. This definition replaces the current
centered, variance-normalized mean-log-mel energy everywhere Beetle represents
N, including Stage 1 targets and Stage 2 acoustic statistics. A single energy
definition is required so FeatureLinear targets and later conditioning remain
compatible.

## Stage 1 data flow

For every Stage 1 batch:

```text
encoder mel -> sampled posterior -> FeatureLinear -> predicted F0/N
target mel  -> frozen F0 extractor              -> ground-truth F0
target mel  -> StyleTTS2 log_norm               -> ground-truth N

posterior + ground-truth F0/N -> decoder -> generator -> waveform
predicted F0/N vs ground truth -> supervised FeatureLinear losses
```

The discriminator and generator passes both use ground-truth F0/N. Stage 1
validation uses the same ground-truth conditioning, matching StyleTTS2 first
stage validation. Validation artifacts continue to plot ground truth against
FeatureLinear predictions so prediction quality remains visible.

Stages 2 and 3 continue using predicted acoustic features because their purpose
is conditional synthesis without ground-truth acoustic tracks.

## Reconstruction weight

The shared `losses.reconstruction.value` changes from 5 to 45. The default
configuration deliberately reuses this loss mapping across stages, so every
stage that performs waveform reconstruction receives the requested weight.
This matches the successful direct iSTFTNet2-MB training signal.

## Posterior diagnostics

At the existing 250-step diagnostic interval, report valid-latent statistics:

- `posterior/log_scale_mean`
- `posterior/log_scale_min`
- `posterior/log_scale_max`
- `posterior/noise_scale_mean`

The last metric is `mean(exp(log_scale))` and makes variance collapse directly
visible. Metrics exclude padded latent positions. No KL behavior changes.

## FeatureLinear clipping

Current evidence shows FeatureLinear exceeds norm 10 on 91.9% of recorded
steps, compared with 42.9% for the audio encoder and 4.3% for the decoder.
FeatureLinear therefore does not receive occasional spike protection; it is
subject to near-continuous rescaling.

Gradient groups gain an explicit clip/observe policy. FeatureLinear uses
observe-only mode: its raw gradient norm is still logged, but its gradients are
not modified. Every other existing group remains clipped at its configured
maximum. This changes only the demonstrated over-clipping problem and does not
remove Beetle's broader stability controls.

## Checkpoints and running jobs

These changes do not mutate an already-running Python process. A restarted
process will load the new behavior. Resuming the current checkpoint would
change its objective and conditioning mid-run, so a clean Stage 1 run is the
valid comparison. The checkpoint format and model parameter shapes do not
change.

## Verification

Use temporary tests, removed before handoff, to establish:

1. Beetle energy equals the algebraically equivalent StyleTTS2 `log_norm` result.
2. Ground-truth F0/N reach the Stage 1 decoder while FeatureLinear predictions
   still receive supervised gradients.
3. Posterior metrics appear only at multiples of 250 and ignore padding.
4. Observe-only FeatureLinear gradients are unchanged; clipped groups retain
   the configured cap.
5. The default configuration parses with Stage 1 reconstruction weight 45.

Run relevant Python checks through `nix develop --command python -m pytest`.
No committed tests or temporary artifacts remain afterward.
