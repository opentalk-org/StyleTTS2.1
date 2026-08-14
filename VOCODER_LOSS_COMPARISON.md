# Vocoder loss comparison

This table compares the waveform objective in this repository with the released StyleTTS-ZS implementation and representative modern vocoders. Flow-based vocoders are included, but their primary objectives are not directly comparable to reconstruction-plus-GAN systems.

| Model | Generation family | Primary reconstruction or generative objective | Spectral supervision | Adversarial objective | Discriminators | Feature/perceptual supervision | Relationship to this Studio implementation |
|---|---|---|---|---|---|---|---|
| **StyleTTS Studio (this repository)** | End-to-end TTS with HiFi-GAN or iSTFTNet waveform decoder | Three-resolution normalized log-mel spectral convergence; average of `L1(error) / L1(target)` | Mel transforms at FFT/hop/window `1024/120/600`, `2048/240/1200`, and `512/50/240` | LSGAN plus TPRLS (`tau=0.04`) | MPD and multi-resolution spectrogram discriminator; optional WavLM discriminator | Discriminator feature matching, WavLM feature loss, WeSpeaker intermediate-feature matching, optional speaker cosine loss | Closely follows released StyleTTS-ZS acoustic loss; adds configurable stages and optional objectives |
| **StyleTTS-ZS official repository** | End-to-end TTS with waveform decoder | Three-resolution normalized log-mel spectral convergence; average of `L1(error) / L1(target)` | Same three FFT/hop/window resolutions as Studio | LSGAN; repository loss stack also includes relative/TPR-style terms | MPD, multi-resolution spectral discriminator, multimodal WavLM discriminator | Discriminator feature matching, WavLM losses, speaker-verifier cosine and intermediate-feature losses | Direct reference implementation; Studio's reconstruction loss is effectively the same, despite the paper describing direct mel L1 |
| **BigVGAN-v2** | GAN mel vocoder | Seven-scale log-mel mean L1 sum | Windows `32-2048`, hops `window/4`, mel bins `5-320`; log magnitude by default | LSGAN | MPD plus multi-scale sub-band CQT discriminator or multi-band discriminator | Discriminator feature matching with effective weight 2 | Broader frequency/time-scale coverage; no division by target norm; strongest direct alternative for a modern GAN ablation |
| **CosyVoice2/3 HiFT** | NSF/iSTFT GAN vocoder | Multi-resolution mel reconstruction | Multiple mel transforms configured by the training recipe | LSGAN plus TPR loss (`tau=0.04`) | HiFi-GAN/HiFT discriminator stack | Discriminator feature matching | Closest modern loss family: multi-mel + feature matching + TPR; differs in mel reduction/normalization and decoder architecture |
| **APNet2** | Direct amplitude-and-phase spectral GAN vocoder | Explicit amplitude, phase, complex-spectrum, STFT-consistency, and mel objectives | Direct supervision of predicted spectral amplitude and wrapped phase | Adversarial waveform/spectral objectives | Multi-period and spectral discriminator variants | Feature matching and spectral consistency terms | Adds explicit phase supervision missing from Studio; especially relevant to an iSTFT decoder experiment |
| **PeriodWave** | Conditional waveform flow matching | Period-aware conditional flow-matching vector-field objective | Mel spectrogram is conditioning; multi-band version models frequency bands separately | None as the primary published objective | No conventional GAN discriminator in the core method | Energy-based prior and period-aware/multi-period estimator | Fundamentally different training family; compare output quality, robustness, and sampling cost rather than mel-loss weights |
| **RFWave** | Rectified flow in time-frequency representation | Rectified-flow velocity regression | Operates on Fourier/spectral representation with auxiliary spectral constraints | None in the core flow objective | No conventional GAN discriminator in the core method | Spectral auxiliary losses depending on recipe | Candidate replacement vocoder, not a drop-in loss change for the current decoder |
| **WaveFM** | Waveform flow matching | Conditional waveform flow-matching regression | Mel/acoustic representation is conditioning rather than the main reconstruction target | None in the core method | No conventional GAN discriminator in the core method | Model-specific auxiliary conditioning losses | Candidate generative-family replacement; requires a new decoder and sampler rather than modifying `mel` loss |
| **DisCoder** | Codec-informed adversarial mel vocoder | Waveform/spectral reconstruction informed by neural-codec representations | Multi-scale spectral supervision | GAN objective | Codec-informed discriminator/encoder-decoder stack | Learned codec feature supervision | Modern learned-perceptual alternative to WavLM/speaker features; materially larger architectural change |
| **DegVoC** | Degradation-aware neural vocoder | Data-prediction/consistency objective combined with mel reconstruction | Mel and degradation/consistency terms | Optional adversarial objective in reported ablations | Vocoder discriminator stack | Consistency supervision | Useful 2025 research comparison for robustness; released implementation details should be verified before porting |
| **VNet** | GAN vocoder with multi-tier discrimination | Mel reconstruction plus adversarial training | Full-band mel information | Asymptotically constrained adversarial loss | Multi-tier discriminator | Feature matching according to its training recipe | Relevant primarily as a discriminator/loss replacement experiment |

## Historical baselines

| Model | Reconstruction | GAN formulation | Discriminators | Main distinction from Studio |
|---|---|---|---|---|
| HiFi-GAN | Single-resolution log-mel mean L1, weight 45 | LSGAN | MPD + MSD | One mel scale, no target-norm division, no TPR or learned speech/speaker losses |
| BigVGAN-v1 | Single-resolution log-mel mean L1, usually weight 45 | LSGAN | MPD + MRD | Predecessor of BigVGAN-v2's multi-scale mel/CQT recipe |
| Vocos | Single-resolution log-mel mean L1, weight 45 | Hinge | MPD + MRD | Fourier-domain generator and hinge GAN |
| DAC | Multi-scale STFT or mel L1 over log and raw magnitudes | LSGAN | Multi-period discriminator | Supervises both raw and log magnitude without target-norm division |
| UnivNet | Multi-resolution STFT reconstruction | LSGAN | Multi-resolution spectrogram discriminator | Established the MRD/spectral-loss lineage used by later vocoders |
| iSTFTNet/iSTFTNet2 | Typically inherited HiFi-GAN-style mel, GAN, and feature-matching losses | LSGAN | MPD/MSD-style stack | Changes waveform synthesis to predicted spectra plus iSTFT more than it changes the objective |

## Most useful controlled comparisons for this repository

| Experiment | Keep fixed | Change | Question answered |
|---|---|---|---|
| Baseline | Current decoder, critics, and all auxiliary losses | Nothing | Establish current validation and listening baseline |
| Remove target normalization | Everything except spectral reduction | Replace `L1(error) / L1(target)` with mean log-mel L1 at the same three resolutions | Does relative normalization improve quiet segments or destabilize level balance? |
| CosyVoice-style | Current decoder and discriminator topology | Multi-mel mean L1 plus TPR and feature matching | Is the current normalized loss better than the closest modern production recipe? |
| BigVGAN-v2-style spectral loss | Current decoder and auxiliary losses | Seven-scale log-mel L1 | Do broader transient/harmonic scales improve fidelity and robustness? |
| BigVGAN-v2 discriminator | Current generator and reconstruction loss | Replace MRD with multi-scale sub-band CQT discriminator | Does frequency-aware adversarial supervision reduce high-frequency artifacts? |
| APNet2-style phase terms | Prefer the iSTFT decoder; retain dataset and evaluation | Add explicit amplitude, phase, and STFT-consistency supervision | Does direct phase supervision benefit the iSTFT synthesis path? |
| Remove learned perceptual terms | Current reconstruction and waveform GAN | Disable WavLM and speaker losses independently | How much quality and speaker identity come from each learned objective? |
| Flow-vocoder benchmark | Same train/validation audio and mel conditioning | Train PeriodWave/RFWave-class decoder separately | Does a modern flow vocoder outperform the GAN decoder at acceptable sampling cost? |

## Interpretation notes

- Loss coefficients cannot be compared directly unless the reductions and observed gradient magnitudes are also compared. A coefficient of `5` on normalized spectral convergence is not inherently weaker than `45` on mean mel L1.
- StyleTTS-ZS and this Studio compute mel spectrograms from generated and target waveforms. The name `MultiResolutionSTFTLoss` is misleading because the actual comparison is performed in log-mel space.
- BigVGAN-v2 and CosyVoice HiFT are the most actionable modern comparisons for the existing GAN decoder.
- APNet2 is the most relevant source of loss ideas for the optional iSTFT decoder.
- PeriodWave, RFWave, and WaveFM require architectural and sampling changes; they are not merely alternative loss functions.
- Entries for rapidly changing or incompletely released projects should be rechecked against the exact Git commit selected for implementation.

## Sources

- [StyleTTS-ZS repository](https://github.com/yl4579/StyleTTS-ZS)
- [StyleTTS-ZS paper](https://arxiv.org/pdf/2409.10058)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [APNet2](https://github.com/redmist328/APNet2)
- [PeriodWave](https://github.com/sh-lee-prml/PeriodWave)
- [RFWave](https://github.com/bfs18/rfwave)
- [WaveFM](https://github.com/luotianze666/WaveFM)
- [DisCoder](https://github.com/ETH-DISCO/discoder)
- [VNet paper](https://arxiv.org/abs/2408.06906)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Vocos](https://github.com/gemelo-ai/vocos)
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
