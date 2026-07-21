# Beetle Stage 1 audio-quality analysis

## Compared runs

- Direct iSTFTNet2-MB baseline: MLflow experiment 1, run
  `02f540e7062747a7bdeb54a748affb4b`, step 6928.
- Beetle Stage 1: MLflow experiment 4, run
  `97cc47b91b484922bdfb64fea1894ccf`, step 8000.

The direct baseline sounds materially better. This is not an exact
architecture comparison: the baseline reconstructs waveform directly from the
ground-truth mel, while Beetle reconstructs through its complete Stage 1
acoustic path.

## Observed evidence

The audio was measured consistently with the same STFT and mel analysis,
independently of each trainer's reported loss.

| Measurement | Direct baseline | Beetle |
| --- | ---: | ---: |
| 0-8 kHz log-mel MAE | 0.469 | 0.593 |
| 1-4 kHz log-magnitude correlation | 0.835 | 0.797 |
| 4-8 kHz log-magnitude correlation | 0.856 | 0.814 |
| 8-12 kHz log-magnitude correlation | 0.767 | 0.819 |
| 4-8 kHz energy error | -4.35 dB | -0.52 dB |
| 8-12 kHz energy error | -6.77 dB | +0.50 dB |

Beetle's 0-8 kHz mel error is approximately 27% worse. However, Beetle
preserves more upper-band energy and has better 8-12 kHz structural
correlation. The audible problem therefore is not simply missing high
frequencies. The evidence points to less faithful harmonic and formant
structure in the speech-dominant 1-8 kHz region.

Waveform sample correlation is not useful here because phase differences make
two perceptually similar reconstructions correlate poorly. Spectral flatness
and low-target-energy measurements also did not support a simple broadband
high-frequency-noise explanation.

## Likely problems

These are ranked investigation targets, not confirmed root causes.

### 1. Reconstruction is underpowered relative to adversarial losses

The successful direct vocoder uses:

```text
generator = adversarial + 2 * feature_matching + 45 * mel
```

At Beetle step 8000, the weighted scalar contributions were approximately:

```text
reconstruction       1.67
F0                   1.99
N                    0.09
adversarial          2.09
feature matching     3.29
```

Direct waveform reconstruction accounted for only about 18% of the Beetle
generator objective. Later gradient diagnostics made the imbalance clearer:

```text
reconstruction waveform gradient      2.84
adversarial waveform gradient         11.09
feature-matching waveform gradient    14.81
```

Adversarial and feature-matching gradients together were approximately 9.1
times the reconstruction gradient. This can favor plausible discriminator
features without preserving the exact acoustic structure of the input.

Beetle also uses normalized spectral convergence on transformed power mels:

```text
sum(abs(target_mel - prediction_mel)) / sum(abs(target_mel))
```

The direct baseline uses ordinary mean L1 on log-magnitude mels. The numeric
loss values are therefore not directly comparable, and Beetle's formulation
may provide a weaker perceptual reconstruction signal at its current weight.

### 2. The sampled posterior is unconstrained

Beetle always reconstructs from:

```text
latent = mean + noise * exp(log_scale)
```

The compared run has an encoder KL weight of zero, while its validation KL is
approximately 100.8. This means the posterior distribution is not trained
toward the standard-normal prior even though a sample from that distribution
is used for reconstruction and later as a Stage 2 target.

The high KL does not prove that posterior noise is degrading the audio; it can
also be caused by informative, large posterior means. A mean-versus-sampled
reconstruction from one checkpoint will determine whether the sampled
variance is audibly harmful.

Validation derives its latent-noise seed from the checkpoint step. Consequently,
different validation steps use different posterior samples. The harmonic
source seed also changes with the step. This adds random variation to
checkpoint-to-checkpoint listening comparisons.

### 3. Predicted F0 errors are amplified by the harmonic source

The direct baseline is conditioned only on the ground-truth mel. Beetle uses:

```text
mel -> sampled posterior -> FeatureLinear -> predicted F0 and N
    -> decoder -> harmonic source -> iSTFT generator
```

The inspected F0 artifact follows the target broadly, but has visible errors
at voiced/unvoiced boundaries and during short pitch excursions. These modest
F0 errors can become audible buzz, roughness, or incorrect harmonic placement
because Beetle explicitly generates a harmonic source from predicted F0.

This remains a candidate rather than a demonstrated cause until the same
checkpoint is synthesized with target F0 and N.

### 4. F0/N conditioning differs between training and validation

During training, the decoder randomly smooths F0 using kernels `[0, 3, 7]` and
N using `[0, 3, 7, 15]`. During evaluation, it always uses unsmoothed F0 and N.
The decoder is therefore often trained on smoother conditioning than it
receives during validation. Predicted boundary errors may be more audible on
the unsmoothed evaluation path.

### 5. The conditioning bottleneck is persistently clipped

The recent FeatureLinear raw gradient norm was approximately 27.7 and was
clipped to 10, retaining about 34% of the requested update. FeatureLinear is
the narrow projection from the 192-channel posterior into only F0 and N. Its
repeated clipping, combined with dominant adversarial gradients downstream,
may slow correction of inaccurate acoustic conditioning.

Raw gradient norms are not directly comparable between Beetle and the direct
vocoder, so disabling all clipping is not justified by this observation.

### 6. The complete Beetle path is much harder than the baseline

The working baseline establishes that the iSTFTNet2-MB synthesis geometry can
produce good audio. It does not establish that Beetle's sampled posterior,
FeatureLinear, decoder, predicted F0/N, and harmonic-source path are correct.
The evidence currently localizes the quality loss upstream of, or at the input
to, the shared synthesis core rather than in the iSTFT/PQMF geometry itself.

### 7. Upper voiced harmonics appear smeared, but not globally more aperiodic

An additional repeated listening and spectrogram observation is that Beetle
often preserves valid lower formants while higher voiced harmonic lines become
diffuse and noise-like. This agrees with the earlier result that Beetle has
approximately correct upper-band energy but worse 1-8 kHz structure.

The initial hypothesis was that the source's fundamental plus eight overtones
created a failure boundary at `9 * F0`. A harmonic-to-interharmonic contrast
measurement rejected this as the primary explanation:

| Harmonic orders | Direct baseline delta | Beetle delta |
| --- | ---: | ---: |
| 2-8 | -3.86 dB | -4.14 dB |
| 10-16 | -3.51 dB | -3.67 dB |
| 17-32 | -1.23 dB | -1.27 dB |

The good direct baseline and Beetle lose nearly the same contrast above the
ninth harmonic. There is no Beetle-specific discontinuity at the source's
highest explicit harmonic.

WORLD bandwise aperiodicity also did not show Beetle becoming uniquely noisier.
For 3-5 kHz, predicted-minus-target aperiodicity was `+0.110` for the direct
baseline and `+0.076` for Beetle. For 5-8 kHz it was `+0.040` and `-0.021`,
respectively. These aggregates do not invalidate a local artifact in particular
phonemes, but they reject a global broadband-unvoiced-noise explanation.

The more accurate description is therefore upper-harmonic smearing or unstable
harmonic placement. Beetle's magnitude-only mel reconstruction can match energy
without directly requiring phase-coherent narrow lines. The spectral
discriminators also operate on magnitude STFTs, and the multi-period path can be
dominated by stronger lower harmonics. Weak reconstruction gradients make this
failure comparatively inexpensive.

Target-F0 and posterior-mean ablations remain the best localization tests. A
complex-STFT, instantaneous-frequency, or harmonic-track-continuity metric would
measure this symptom more directly than total band energy or spectral flatness.

## Differences from vendored StyleTTS2

Beetle reproduces individual StyleTTS2 ideas, but its Stage 1 training path is
not equivalent to StyleTTS2's first stage.

### Ground-truth acoustic conditioning

StyleTTS2 extracts `F0_real` and `real_norm` from the ground-truth mel and feeds
them directly to its decoder. Beetle predicts F0 and N from a sampled posterior
and feeds those predictions to the decoder and harmonic source. Consequently,
StyleTTS2's decoder does not have Beetle's compounded posterior-sampling and
F0/N-prediction error during first-stage reconstruction.

### Reconstruction pretraining before GAN training

Before `TMA_epoch`, StyleTTS2 sets the generator objective to reconstruction
loss alone and does not update its waveform discriminators. Its reference
configuration starts adversarial/TMA training at epoch 50; the LibriTTS
configuration uses epoch 5.

Beetle begins adversarial warmup at step zero and reaches full adversarial and
feature-matching weight at step 1000. This is a material departure in the
harder sampled-posterior path. The direct iSTFTNet2-MB baseline survives early
GAN training because it has ground-truth mel conditioning and a much stronger
reconstruction contribution; that does not demonstrate that Beetle can.

### Harmonic-source injection

StyleTTS2 also uses eight overtones, so harmonic count is not the important
difference by itself. Its iSTFTNet generator transforms the harmonic waveform
to magnitude/phase features, then applies a separate learned source convolution
and residual block at every generator upsampling stage before adding that source
to the main path.

Beetle transforms the same type of source features, projects them once to 64
channels, applies one residual block, and adds them once before its MRF and
frequency-expansion path. This may make periodic information easier to wash out,
but the current artifacts do not isolate it from posterior/F0 and loss-balance
effects. It ranks below the target-F0 and posterior-mean tests.

### Reconstruction formulation

Vendored StyleTTS2 uses the same three power-mel resolutions, normalized log-mel
spectral-convergence formula, and nominal mel weight of 5 after adversarial
training begins. Therefore Beetle's loss formula is faithful to StyleTTS2 in
isolation. The relevant difference is its schedule and the acoustic information
presented to the decoder, not merely the number 5.

## Recommended diagnostic experiment

Do not restart training to localize this. Export the same validation recordings
from the same checkpoint under four conditions:

1. Sampled posterior with predicted F0/N, matching current validation.
2. Posterior mean with predicted F0/N.
3. Sampled posterior with target F0/N.
4. Posterior mean with target F0/N.

Use an identical harmonic-source seed in all four conditions. Record the same
consistent 0-8 kHz mel error and band correlations for each output. Also record
harmonic-track continuity or instantaneous-frequency error; the completed
aperiodicity and `9 * F0` checks were not discriminating.

Interpretation:

- If condition 2 improves, posterior sampling or posterior variance is harmful.
- If condition 3 improves, FeatureLinear/F0/N conditioning is the main defect.
- If condition 4 alone improves, both paths contribute.
- If none improves, focus next on loss balance, decoder features, and the
  harmonic-source implementation.

After localization, the most justified training ablation is a checkpoint copy
with stronger direct reconstruction and delayed or reduced adversarial and
feature-matching weights. A second focused ablation can raise only the
FeatureLinear gradient cap. Neither change should be applied blindly to the
current run before the four-way export.
