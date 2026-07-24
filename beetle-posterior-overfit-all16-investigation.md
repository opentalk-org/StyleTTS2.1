# Beetle All-16 Posterior Overfit Investigation

## Dataset and validation scope

Dataset `1010220c-5eaa-44f1-983f-4d7123ad1564` contains all 16 original
validation recordings, each copied four times. The 64 rows satisfy the
unchanged GE2E sampler requirement of at least two utterance records per
voice. Batch size remains 32.

The prior random validation selector covered only 10 unique source paths.
Validation now explicitly selects one copy of every source, so training sees
all 64 rows and validation covers exactly all 16 unique recordings.

## Teacher-forced acoustic conditioning

Run `3066df4a70e8465c816ec403192b98af` used ground-truth F0 and N in the
training decoder path:

- steps 950–999 reconstruction: mean `0.357470`, standard deviation
  `0.036325`, minimum `0.284086`, maximum `0.441664`;
- step-1,000 validation reconstruction: `0.386400`;
- ground-truth-F0/N full reconstruction at checkpoint step 1,170: `0.338629`.

The configured acoustic prediction ratio was zero through step 2,000, while
validation and inference always used F0 and N predicted from the posterior
latent. Holding latent and source noise fixed at checkpoint step 1,170 gave:

| Predicted F0/N ratio | Reconstruction |
| ---: | ---: |
| 0.00 | `0.338629` |
| 0.25 | `0.345812` |
| 0.50 | `0.351057` |
| 0.75 | `0.358828` |
| 1.00 | `0.380453` |

The monotonic `0.041824` penalty proves the train/inference conditioning
mismatch. Exhaustive configured smoothing-kernel tests improved `0.380453`
only to `0.377097`, so smoothing is secondary.

Component replacement localized the penalty:

| F0 source | N source | Reconstruction |
| --- | --- | ---: |
| Ground truth | Ground truth | `0.338629` |
| Ground truth | Predicted | `0.343124` |
| Predicted | Ground truth | `0.375769` |
| Predicted | Predicted | `0.380453` |

Predicted F0 accounts for approximately `0.0373` of the total `0.0418`
conditioning penalty; predicted N accounts for approximately `0.0045`.

## Matched predicted-conditioning run

Run `7aa3844c0c4940eaac890506c29ca793` used predicted F0 and N from step zero:

- steps 950–999 reconstruction: mean `0.369173`, standard deviation
  `0.036846`, minimum `0.297130`, maximum `0.449694`;
- step-1,000 validation reconstruction: `0.363337`;
- validation F0 loss: `16.7968`;
- validation N loss: `0.35635`.

The validation-minus-training gap changed from `+0.02893` with teacher forcing
to `-0.00584` with matched predicted conditioning. The pipeline mismatch was
removed, but output quality remained poor.

## Spectral inspection

The saved STFT images show correct coarse timing but nearly absent harmonic
ridges, broad low-frequency blobs, vertical comb texture, and missing upper
harmonics.

The original validation `mel.png` is misleading because its ground-truth panel
uses the padded merged batch mel while the posterior panel uses a trimmed
waveform mel. The panels therefore have different time extents and independent
color scales. Recomputing equal-length panels with a shared scale still shows
the same real failure: smeared energy, weak harmonic contrast, and suppressed
high-frequency detail. The poor mel is not merely a plotting artifact.

## F0 output scale

The F0 head previously applied `softplus` directly to an unscaled projection,
so a zero logit represented `0.693 Hz`. Across the 16 validation sources at
the step-1,170 checkpoint:

- target voiced F0: mean `177.133 Hz`, standard deviation `96.94`;
- predicted F0: mean `21.164 Hz`, standard deviation `38.56`, maximum
  `267.84 Hz`, correlation `0.197`.

A frozen-latent probe compared identical zero-initialized 1x1 heads under the
same Adam learning rate. With unit output scale, the F0 loss moved from
`17.5978` to `13.9792` by step 1,000 and mean prediction reached only
`63.380 Hz`. With a `256 Hz` output scale, loss moved from `7.6971` to
`5.4306`, mean prediction was `179.079 Hz`, and correlation reached `0.6377`.

This isolates an output parameterization problem rather than insufficient
capacity. The production head now:

- requires `feature.f0_scale_hz` in configuration;
- uses `256.0` in the current config;
- zero-initializes only the F0 projection row;
- computes `softplus(logit) * f0_scale_hz`.

A focused test verifies the zero-initialized value is `log(2) * 256 Hz` and
that frame masking is preserved.

## Storage incident

Cancellation of the matched predicted-conditioning run could not save a
checkpoint because the filesystem was full. Its incomplete temporary
checkpoint was removed. Checkpoint folders from eight documented,
reverted/superseded ablations were also removed; their MLflow metrics and logs
remain. Approximately 22 GB was recovered.

The next from-zero run tests only the F0 output-scale correction with predicted
F0/N conditioning active from step zero.

## Scaled-F0 run and voiced-spectrum localization

Run `23f8ada274764f63a5bbadff54c8fa16` tested the `256 Hz` F0 scale:

- steps 950–999 reconstruction: mean `0.377421`, standard deviation
  `0.035707`, minimum `0.313360`, maximum `0.463705`;
- step-1,000 validation reconstruction: `0.373781`;
- step-1,000 validation F0: `6.23157`, versus `16.7968` before scaling;
- step-1,000 validation N: `0.662782`, versus `0.35635` before scaling.

The scale correction made voiced F0 contours substantially more accurate but
did not improve reconstruction. The feature-head gradient norm at step 1,000
grew from `3.48` to `85.22`. The F0 loss must be computed in scale-normalized
units rather than allowing the `256` output scale to multiply feature-head
parameter gradients.

The optimizer-level clipping coefficient is only the minimum coefficient
reported across named gradient groups. Clipping is applied independently to
each group, so neither the F0 head nor latent flow rescales every generator
parameter. Earlier notes describing optimizer-wide gradient suppression do not
match the current implementation.

The F0 plot also exposed missing voicing supervision. Ground-truth F0 contains
exact zeros, but the loss selected only target-voiced frames and the positive
`softplus` output remained around `100–200 Hz` through unvoiced intervals.
This keeps harmonic excitation active and explains comb energy in target-dark
regions. The active untrained correction adds a separate sigmoid voicing gate,
supervises the gated normalized F0 on every valid frame, and retains the
learnable voiced-frequency magnitude.

Neither problem explains the poor voiced regions. Re-rendering the retained
step-1,170 checkpoint with both ground-truth F0 and N produced essentially the
same broad low-frequency components and regular upper-frequency comb. The
voiced failure is downstream of acoustic control prediction.

The validation mel artifact bug is also corrected in the active code.
Ground-truth and prediction mels are now computed per sample after slicing both
waveforms to the same true length. This removes the former padded
`3000`-frame ground-truth panel versus approximately `650`-frame prediction
panel mismatch.

## Frequency-network capacity probes

One `19,200`-sample voiced crop was used to separate synthesis representation
from the learned frequency mapping. Directly optimizing the complex spectra
through the unchanged four-band iSTFT and PQMF backend gave:

| Step | Reconstruction | Waveform L1 |
| ---: | ---: | ---: |
| 0 | `2.63868` | `0.046464` |
| 100 | `0.12607` | `0.036841` |
| 250 | `0.03403` | `0.008337` |
| 500 | `0.01874` | `0.002334` |

The resulting STFT visually retains the target harmonic ridges. PQMF, iSTFT,
and the subband spectrum geometry therefore have sufficient representational
capacity.

The step-1,170 generator frequency network was then frozen while its entire
`64 x time` input was optimized freely. Reconstruction reached only `0.33609`
at step 500 and waveform L1 remained `0.04792`. Jointly optimizing the
frequency network and its free input improved reconstruction to `0.12418`,
but waveform L1 remained `0.05154`.

The same crop's multi-resolution linear-STFT diagnostics were:

| Output | Spectral convergence | Log-magnitude L1 | Waveform L1 |
| --- | ---: | ---: | ---: |
| Ground-truth-F0/N model output | `0.82086` | `0.17208` | `0.05913` |
| Direct spectrum oracle | `0.02125` | `0.00822` | `0.00233` |
| Trainable frequency-network probe | `0.68226` | `0.12734` | `0.05153` |

This localizes the voiced blur to the learned temporal-to-frequency network
and its supervision. It does not justify enlarging the whole decoder or
changing PQMF.

With the same trainable frequency network and free temporal input, replacing
the mel objective with direct multi-resolution linear-STFT magnitude
supervision gave:

| Step | Spectral convergence | Log-magnitude L1 | Waveform L1 |
| ---: | ---: | ---: | ---: |
| 0 | `0.99941` | `0.26257` | `0.04647` |
| 100 | `0.43392` | `0.10010` | `0.05998` |
| 250 | `0.21406` | `0.06126` | `0.05826` |
| 500 | `0.09135` | `0.03333` | `0.05340` |

The resulting magnitude STFT recovers sharp curved harmonic ridges. This
proves the existing frequency architecture can express the missing voiced
detail when directly supervised. Waveform L1 does not improve because this
diagnostic intentionally constrains magnitude rather than phase.

The active production ablation therefore trains reconstruction with the mean
of three linear-STFT spectral-convergence plus log-magnitude losses. The prior
mel reconstruction remains separately computed and logged under the original
`posterior_reconstruction` name, preserving comparison to the `< 0.35`
criterion. The optimized spectral objective is logged as
`posterior_spectral_reconstruction`.

The first full run replaced mel supervision with the spectral objective. It
was stopped at step 300 because its last-50 mel reconstruction at step 261 was
`0.76669`, versus approximately `0.48` on the matched mel-trained trajectory.
The spectral metric was still `0.93320`. Replacing mel had reduced its measured
waveform-gradient norm by approximately `19.5x`.

The second full run retained mel at weight `45` and initially set spectral
weight `900` from that waveform-gradient ratio. At step 100, however, the
generator parameter-gradient norm was `703`, versus `109` on the matched
baseline; audio-encoder norm was `191`, versus `7.3`. The routing allowed
spectral supervision to dominate the upstream posterior and decoder rather
than specializing the waveform generator. It was stopped at step 162.

The active routing keeps mel, adversarial, and feature-matching gradients
through the complete posterior stack. Spectral loss backpropagates only to
waveform-generator parameters. A focused gradient test verifies that selective
backward leaves upstream gradients equal to the ordinary objective while
generator gradients contain both objectives. The spectral weight is `150`,
estimated from the approximately `6.45x` generator-parameter norm increase
observed at weight `900`.

## Native-head step-2,064 dissection

The retained native-frequency checkpoint was dissected on validation sample 16.
Posterior noise is no longer material: mean posterior scale is `0.0244`, and
sampled versus posterior-mean mel is `0.27874` versus `0.27869`.

Predicted F0 remains inaccurate (`0.257` voiced correlation and `48.8 Hz`
voiced MAE), but this is not the immediate reconstruction bottleneck.
Ground-truth F0/N worsens mel to `0.29690` and barely changes ridge recovery.
Replacing only harmonic-source F0 changes the waveform by `1.7%`; zeroing it
changes the waveform by `2.3%`. The temporal harmonic projection has RMS
`0.264` against `8.80` for temporal generator features. The current generator
has therefore learned to mostly ignore its harmonic source.

Frequency structure is progressively suppressed:

| Location | RMS | normalized frequency curvature |
| --- | ---: | ---: |
| frequency entry | `24.44` | `2.639` |
| final frequency upsample | `0.225` | `2.080` |
| native refinement | `0.278` | `1.401` |
| band 0 output | `0.326` | `0.911` |
| band 3 output | `0.191` | `0.152` |

Magnitude is the collapsed channel. Band-0 log-magnitude frequency curvature
is `0.511`; band 3 is `0.087`. Phase curvature remains `2.77` and `2.08`.
Random initialization is even flatter at the heads, so the architecture starts
with a strong frequency-flat bias; training recovers low-band detail but not
upper-band magnitude detail.

This is not caused by missing high-band gradients. The 8–12 kHz region is
`10.6%` of target normalization and contributes `0.0306` to sample mel loss.
Its magnitude-gradient RMS on band head 3 is `5.04e-4`, versus `3.19e-4` from
0–1 kHz on band head 0. Bandwise gradient cosines are approximately zero, so
the bands are not fighting each other either.

A frozen-feature, band-head-only probe separates capacity from objective:

| Objective, step 200 | mel | 4–8 kHz ridge | 8–12 kHz ridge |
| --- | ---: | ---: | ---: |
| checkpoint | `0.27874` | `0.0159` | `0.0188` |
| mel only | `0.26262` | `0.0123` | `0.0186` |
| high-frequency curvature only | `1.20438` | `0.0265` | `0.0490` |
| mel + `0.05` curvature | `0.26470` | `0.0357` | `0.0737` |

The same frozen heads can therefore produce much sharper upper harmonics, and
a small curvature term improves mel and all four ridge bands simultaneously.
Ordinary pointwise mel optimization explicitly improves its scalar while
making 4–8 kHz ridge recovery worse. The primary remaining sharpness failure
is objective mismatch, compounded by an underpowered harmonic prior and
frequency-flat head initialization—not posterior sampling, PQMF/iSTFT
capacity, absent gradients, or F0 accuracy alone.
