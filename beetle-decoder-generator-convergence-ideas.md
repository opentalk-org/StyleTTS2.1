# Beetle Decoder/Generator Convergence Ideas

## 1. Separate latent-flow optimization from the acoustic generator

This is the highest-priority change because latent flow and the acoustic
generator currently share optimizer-wide gradient clipping.

Observed examples:

- At step 400, the total generator-group gradient norm was approximately
  `28,228`, the latent-flow norm was `28,220`, and the waveform-generator norm
  was `208`.
- The resulting global clip coefficient was approximately
  `10 / 28,228 = 0.000354`.
- This reduced the waveform generator's effective gradient norm from `208` to
  approximately `0.074`.
- At step 2,000, the latent-flow norm was `4,408`, the waveform-generator norm
  was `185`, and the global clip coefficient was `0.00227`, leaving the
  waveform generator with an effective norm of approximately `0.42`.

Latent flow therefore consumes almost the entire shared gradient budget. The
decoder and waveform generator learn despite being heavily suppressed.

Recommended change:

- Give `latent_flow` its own optimizer, AMP scaler, scheduler, and gradient
  clipping.
- Keep the audio encoder, feature projection, decoder, and waveform generator
  in the acoustic optimizer.
- Consider isolating duration flow too, although it is not currently the
  dominant gradient source.
- Retain the acoustic learning rate of `6e-4` initially and tune the flow
  optimizer independently.

This is the cleanest first ablation because it improves optimization without
changing model expressiveness.

## 2. Supervise the representation predicted by the generator

The generator predicts complex subband spectra, but its deterministic
reconstruction objective currently consists of normalized multi-resolution
mel-magnitude L1. Phase and full-resolution linear-frequency structure are
learned mostly through adversarial and feature-matching losses.

Retain the mel loss and add:

- Multi-resolution linear-STFT log-magnitude loss.
- Spectral-convergence loss.
- A small complex-STFT or waveform loss for phase and transient alignment.
- A magnitude mask for the complex loss so arbitrary phase in silent bins does
  not dominate.

This would provide the iSTFT generator with a direct early training signal.
The spectral-loss weight should be warmed up and later reduced as the GAN and
feature-matching objectives become reliable. Too much spectral supervision can
produce smooth but dull audio.

References:

- [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
- [UnivNet](https://arxiv.org/abs/2106.07889)
- [Vocos](https://arxiv.org/abs/2306.00814)

## 3. Replace blind harmonic addition with controlled source-filter fusion

The current fusion is:

```python
temporal = temporal + source
```

Both branches have 64 channels, but their scales are uncontrolled and their
identities disappear after addition. The F0-derived source can become a
shortcut, while the feature branch must learn to cancel or reshape it.

A better source-filter design would:

- Concatenate the feature and harmonic branches instead of adding them.
- Mix them with a learned `1x1` projection or feature-conditioned gate.
- Initialize the source gate to a small value such as `0.05` to `0.1`, making
  early reconstruction feature-driven.
- Inject excitation hierarchically at two or three generator depths.
- Preserve separate periodic and aperiodic excitation channels.

This makes the harmonic branch an explicit excitation prior: it supplies pitch,
while feature-conditioned layers determine resonance and timbre.

References:

- [Source-Filter HiFi-GAN](https://arxiv.org/abs/2210.15533)
- [Unified Source-Filter GAN with Harmonic-plus-Noise Excitation](https://arxiv.org/abs/2205.06053)

## Recommended ablation order

1. Isolate latent-flow gradients.
2. Add direct STFT and phase-sensitive supervision.
3. Test gated source-filter fusion.

Increasing generator size should come afterward. The current generator may
eventually prove too small, but additional capacity will not fix its gradients
being scaled down by the flow model. If capacity remains limiting after
optimizer isolation, increase the 64-channel post-fusion backbone/frequency
path or test anti-aliased SnakeBeta blocks from
[BigVGAN](https://arxiv.org/abs/2206.04658).
