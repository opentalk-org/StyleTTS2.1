# Beetle Decoder/Generator Convergence Ideas

## 1. Optimizer isolation is not required by current clipping

This hypothesis does not match the current optimizer implementation. Gradient
clipping is applied independently to each named module group. The
optimizer-level coefficient is only the minimum group coefficient reported as
a diagnostic; it is not applied to every optimizer parameter.

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

These observations came from interpreting the optimizer-level diagnostic as a
global clip. They do not show latent flow suppressing acoustic updates in the
current code.

Recommended change:

- Give `latent_flow` its own optimizer, AMP scaler, scheduler, and gradient
  clipping.
- Keep the audio encoder, feature projection, decoder, and waveform generator
  in the acoustic optimizer.
- Consider isolating duration flow too, although it is not currently the
  dominant gradient source.
- Retain the acoustic learning rate of `6e-4` initially and tune the flow
  optimizer independently.

Optimizer separation may still be useful for independent schedules, but it is
not a fix for shared clipping because clipping is already independent.

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

## High-frequency smoothing investigation

### Observed failure signature

The posterior reconstruction broadly matches the ground-truth time-frequency
structure, so event placement and the dominant spectral envelope are being
transmitted. The remaining error has a consistent fine-structure signature:

- Harmonic ridges and vertical textures are blurred, particularly around
  frames `20–50`, `110–130`, `180–215`, and `280–305`.
- Strong components spread into adjacent time-frequency bins, producing thicker
  bands and less precise event boundaries.
- Faint spurious energy fills otherwise dark mid- and high-frequency regions.
- Mid-frequency energy around frames `285–300` is redistributed and diffuse.
- Some events leak slightly before or after their target duration, especially
  low-frequency segments near frames `160–170` and `280–325`.
- Fine structures above approximately frequency bin `150` are suppressed and
  unusually uniform.
- Closely spaced horizontal harmonics merge in voiced regions.

The combined symptom is spectral blur, leakage, excess low-level background
energy, and reduced harmonic contrast rather than incorrect placement of the
dominant events.

### Current localization evidence

A forward hook captured the generator tensor immediately before
`MultiBandISTFT` using the latest available checkpoint, whose payload reports
step `8,000`. Adjacent-frequency variation in predicted subband log magnitude
was compared with a PQMF analysis of ground truth:

| PQMF band | Predicted/target spectral contrast |
| --- | ---: |
| 0 | `15.8%` |
| 1 | `16.6%` |
| 2 | `5.1%` |
| 3 | `4.8%` |

An ideal STFT/iSTFT/PQMF round trip retained `99.99%` of full-band mel spectral
contrast. The smoothing is therefore already present in the frequency
network's magnitude prediction; ordinary iSTFT and PQMF synthesis do not create
it.

The phase path applies `sin` to its raw prediction and consequently restricts
the represented phase to `[-1, 1]` radians. The observed range was approximately
`[-0.88, 0.97]`. This deserves an independent phase ablation, but it does not
explain why pre-iSTFT log magnitude is already smooth.

### Architectural hypothesis

PQMF does not inherently require four completely independent generators.
Sharing temporal processing is useful because all bands describe the same
events. The current frequency head is nevertheless unusually restrictive:

```text
4 frequency bins → 8 → 16 → 31
64 channels      → 32 → 16 → 8
```

The final eight channels directly represent magnitude and phase for four PQMF
bands. There is no nonlinear residual refinement after reaching the native
31-bin resolution. This structure favors smoothly interpolated spectra, and
the much larger collapse in bands 2 and 3 suggests that a shared representation
plus a minimal final projection is not adequately modeling the different
statistics of upper PQMF bands.

### Test 1: magnitude/phase oracle decomposition

Run the same fixed samples through four synthesis variants without training:

1. predicted magnitude + predicted phase;
2. predicted magnitude + ground-truth PQMF phase;
3. ground-truth PQMF magnitude + predicted phase;
4. ground-truth magnitude + ground-truth phase.

Bypass the phase parameterization when supplying ground-truth phase. Compare
audio, mel spectral contrast, transient width, and bandwise STFT error.

- Variant 2 remaining blurred implicates magnitude.
- Variant 3 remaining poor implicates phase.
- Both mixed variants improving materially means both paths contribute.
- Variant 4 verifies the analysis/synthesis test fixture.

### Test 2: one-segment capacity tests

Use one fixed segment and disable GAN variability:

1. Optimize directly learnable temporal input features through the generator.
2. Optimize the complete posterior reconstruction stack on the same segment.

Use direct subband magnitude supervision during this diagnostic so the existing
loss cannot hide model capacity.

- If the generator cannot reproduce sharp ridges from learnable features, its
  frequency architecture is the bottleneck.
- If the generator succeeds but the full posterior stack fails, the
  audio-encoder/decoder representation discards required information.
- If both succeed, normal multi-example optimization or loss weighting causes
  the collapse.

### Test 3: supervision-only ablation

Before changing architecture, add an auxiliary target at the representation
already produced by the generator:

1. PQMF-analyze the target waveform.
2. Compute the same per-band STFT used by `MultiBandISTFT`.
3. Apply masked L1 to predicted and target log magnitude.
4. Report the loss, spectral contrast, and curvature separately for every band.
5. Initially omit phase so magnitude and phase conclusions remain separable.

Resume the same checkpoint for approximately `1,000–2,000` steps with fixed
validation samples and seeds.

- Rapid recovery of bands 2–3 means the existing waveform/mel/GAN objectives
  do not adequately supervise internal high-frequency structure.
- A falling auxiliary loss without recovered contrast suggests the metric or
  target construction is wrong.
- A stubborn auxiliary loss in a one-segment run indicates insufficient model
  capacity or poor frequency parameterization.

A full-band spectral-derivative loss is a cheaper alternative, but it is less
diagnostic and may sharpen background noise along with real harmonics.

### Architecture ablations

Test only after the oracle, capacity, and supervision experiments.

#### A. Wider native-resolution refinement

This is the smallest recommended architecture change:

- Preserve more than eight channels after reaching 31 frequency bins.
- Add two or three residual `Conv2d` blocks operating at all 31 bins.
- Project to the final eight magnitude/phase channels only at the output.

This tests whether early channel collapse and the absence of native-resolution
nonlinearity prevent sharp spectral predictions.

#### B. Lightweight band-specific heads

Keep the expensive temporal network shared, then branch into four compact
frequency heads:

```text
shared temporal features
  ├─ band 0 frequency head → magnitude + phase
  ├─ band 1 frequency head → magnitude + phase
  ├─ band 2 frequency head → magnitude + phase
  └─ band 3 frequency head → magnitude + phase
```

Each head should include native-31-bin residual refinement. This allows the
upper bands to learn different statistics without duplicating the complete
generator. A two-head low/high split is a cheaper intermediate test.

#### C. Independent phase parameterization test

Compare the current `sin(raw_phase)` representation with a parameterization
that covers the complete phase circle, while holding magnitude and architecture
fixed. Do not combine this with a magnitude-head change because the oracle test
must first determine whether phase materially contributes to the audible
failure.

### Evaluation and stopping criteria

Use fixed samples, crop locations, seeds, and checkpoint initialization. Track:

- pre-iSTFT spectral contrast and curvature per PQMF band;
- full-band mel contrast in `0–1`, `1–4`, `4–8`, and `8–12 kHz`;
- energy in target-silent time-frequency bins;
- transient/event width and temporal leakage;
- bandwise reconstruction error;
- listening comparisons;
- feature matching and adversarial loss as secondary metrics.

Do not select an ablation using scalar reconstruction loss alone. A successful
change must increase band 2–3 spectral contrast without increasing spurious
background energy or destabilizing temporal boundaries.

Recommended order:

1. magnitude/phase oracle decomposition;
2. one-segment capacity tests;
3. direct subband-log-magnitude supervision;
4. wider native-resolution refinement;
5. low/high or four-way band-specific heads;
6. phase parameterization only if the oracle test implicates phase.
