# Working iSTFTNet2-MB versus Beetle

Reference: `training/istfnet2_mb`. Port: `training/beetle`.

## End-to-end training graph

```text
Working
waveform ── LogMel 0–12 kHz ───────────────┐
         └─ LogMel 0–8 kHz ── frozen JDC ──┼─ Generator ── iSTFT ── waveform
                                            └─ explicit F0 harmonic source

Beetle posterior path
waveform ── LogMel 0–12 kHz ── stochastic posterior encoder ── latent ──┐
         └─ LogMel 0–8 kHz ── frozen JDC ── target F0 ──────────────────┤
latent ── FeatureLinear ── predicted F0/N ── curriculum ────────────────┤
                                                                       └─ Decoder
                                                                          │ 512 features
                                                                          ▼
                                                                    Generator ── iSTFT
```

The Beetle model is not training the same vocoder problem. The working model maps
the target mel directly to audio. Beetle must simultaneously learn a stochastic
audio representation, a decoder, F0/N prediction, the vocoder, and the
conditional model.

## Signal and batch setup

| Part | Working | Beetle | Match |
|---|---|---|---|
| Sample rate | 24 kHz | 24 kHz | Yes |
| Segment | 24,576 samples | 24,576 samples | Yes |
| Conditioning STFT | 1024/256/1024 | 1024/256/1024 | Yes |
| Conditioning mel | 80 bins, 0–12 kHz, Slaney | Same | Yes |
| JDC mel | 80 bins, 0–8 kHz | Same | Yes |
| Batch size | 16 | 16 | Yes |
| Precision | FP32 | FP32 | Yes |
| Examples | Sequential 1.024 s waveform chunks | Random aligned windows from sentence cuts | **No** |
| D/G example in a step | Same generated waveform | Same retained acoustic synthesis | Yes |
| Waveform padding | Mel frames × 256 | Mel frames × configured hop | Yes |

The hardcoded 300 does not shift a segment beginning at frame zero, but violates
the 256-hop geometry and can affect padded full-recording and validation paths.

## Generator

| Layer | Working | Beetle | Match |
|---|---|---|---|
| Input projection | 80 mel → 512 | 512 decoder features → 512 | **No** |
| Upsampling | 512→256→128, rates 8×8, kernels 16×16 | Same | Yes |
| Resblock paths | Kernels 3/7/11, dilations 1/3/5 | Same | Yes |
| Last-stage bottleneck | 2× | 2× | Yes |
| Harmonics | F0 + 8 overtones | Same | Yes |
| Source injection | 18-bin magnitude/phase at both stages | Same geometry | Yes |
| Output | 18 channels: 9 magnitude + 9 phase | Same | Yes |
| iSTFT | FFT 16, hop 4 | Same geometry | Yes |
| Output hop | 8×8×4 = 256 | Same | Yes |

Material implementation differences:

- Source convolutions now retain the same PyTorch default initialization as the
  working model. Weight-normalized layers use the same legacy initialization
  sequence as the reference.
- Working `TorchSTFT.transform` detaches source phase. Beetle's `torch.angle`
  leaves phase differentiable, so the harmonic merge receives an additional
  gradient path.
- Working uses the legacy matrix/convolution STFT implementation; Beetle uses
  `torch.stft`/`torch.istft`. Frame and sample geometry match.
- The working generator has about 15.61 M parameters. Beetle's 512-channel input
  projection adds about 1.55 M parameters compared with the 80-channel input.

## F0 and decoder conditioning

| Part | Working | Beetle | Consequence |
|---|---|---|---|
| F0 during training | Frozen JDC output | JDC until step 1000, then blend to prediction through step 5000 | Match only before transition |
| Unvoiced prediction | JDC can output exact zero | `softplus(magnitude) * sigmoid(voicing)` is almost never zero | Harmonics in predicted unvoiced regions |
| F0 preprocessing | Direct to harmonic source | Direct to harmonic source | Yes |
| N feature | Not used | Ground truth, then blended to prediction with F0 | Additional learned dependency |
| Vocoder features | Target mel | Learned decoder features | Fundamentally different |

Validation renders the posterior reconstruction with ground-truth F0 and N.
Predicted F0 and N are still reported as diagnostics but no longer condition the
validation waveform.

## Discriminators and losses

| Part | Working | Beetle | Match |
|---|---|---|---|
| MPD periods/layers | 2, 3, 5, 7, 11 | Same | Yes |
| MRSD resolutions | (1024,120,600), (2048,240,1200), (512,50,240) | Same | Yes |
| MRSD input | Raw STFT magnitude | Raw STFT magnitude | Yes |
| Discriminator loss | LSGAN + TPRLS | Same formulas | Yes |
| Generator adversarial | LSGAN + TPRLS | Same formulas | Yes |
| Feature matching | Sum L1 × 2 | Same formula | Yes |
| Reconstruction | Slaney log-mel L1 × 45 | Same transform/formula/weight | Yes |
| Extra acoustic losses | None | F0 ×1, N ×1, KL ×0 | **No** |
| Extra joint losses | None | Conditional losses in same generator optimizer | **No** |

The `log1p` in Beetle's spectral discriminator changes both discriminator scale
and gradients. Copying only the loss formulas did not make adversarial training
equivalent.

## Optimizer and schedule

| Part | Working | Beetle |
|---|---|---|
| Generator parameters | Generator only | Encoder, FeatureLinear, decoder, generator, and all conditional modules |
| Optimizer | AdamW | AdamW |
| LR | 2e-4 immediately | Linear warmup 0→2e-4 over 100 steps |
| LR decay | ×0.999 per epoch | Cosine to 2e-5 by step 10k |
| Betas | 0.8, 0.99 | Same |
| Epsilon | AdamW default 1e-8 | 1e-8 |
| Weight decay | AdamW default 0.01 | 0.01 |
| Generator clip | Global 5000 | Global 5000 |
| Discriminator clip | 100 | 100 |
| Numeric execution | FP32 | FP32 |

Per-module gradient norms remain diagnostic-only; clipping is applied once over
the full generator optimizer parameter set.

## Why raw waveform correlation is near zero

The generator is optimized for mel magnitude, discriminator responses, and
feature maps. None fixes absolute waveform phase. Its harmonic source also
contains random initial phase and noise. Saved step-1000 artifacts confirm:

| Metric, mean over 16 samples | Working | Beetle posterior |
|---|---:|---:|
| Raw waveform correlation | 0.0024 | 0.0062 |
| Best correlation within ±256 samples | 0.0385 | 0.0523 |
| Hilbert-envelope correlation | 0.4154 | 0.5269 |
| Target RMS | 0.0700 | 0.0692 |
| Prediction RMS | 0.0400 | 0.0518 |

Raw correlation therefore does not explain the audible difference and should
not be used as the quality criterion for these runs.

## Highest-priority equivalence gaps

1. Treat direct-mel iSTFTNet and Beetle's learned-feature vocoder as different
   experiments. Exact step-for-step quality parity is not expected while the
   upstream posterior and decoder are trained from scratch.
