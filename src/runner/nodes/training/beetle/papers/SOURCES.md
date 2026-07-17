# Beetle Research Sources

Retrieved 2026-07-17. Repository snapshots are shallow clones kept as local
implementation evidence. They are not imported by Beetle at runtime.

## Repositories

| Project | URL | Source commit | License | Purpose |
| --- | --- | --- | --- | --- |
| StyleTTS2 | https://github.com/yl4579/StyleTTS2 | `5cedc71c333f8d8b8551ca59378bdcc7af4c9529` | MIT | ALBERT, aligner, F0, GAN, and acoustic-loss reference |
| Piper | https://github.com/rhasspy/piper | `73c04d81d5590ecc46e522de3601ce7fb29fc2be` | MIT | VITS posterior encoder and stochastic duration-flow reference |

## Papers

| Local folder | arXiv | Title | Used for |
| --- | --- | --- | --- |
| `styletts2/` | [2306.07691](https://arxiv.org/abs/2306.07691) | StyleTTS 2 | Existing model, aligner, F0, and discriminator behavior |
| `vits/` | [2106.06103](https://arxiv.org/abs/2106.06103) | VITS | Posterior encoder and stochastic duration predictor |
| `flow-matching/` | [2210.02747](https://arxiv.org/abs/2210.02747) | Flow Matching for Generative Modeling | Conditional flow-matching objective |
| `diffusion-forcing/` | [2407.01392](https://arxiv.org/abs/2407.01392) | Diffusion Forcing | Independent per-token noise levels |
| `shortcut-models/` | [2410.12557](https://arxiv.org/abs/2410.12557) | One Step Diffusion via Shortcut Models | Dyadic step conditioning and EMA bootstrap |
| `istftnet2-mb/` | [2308.07117](https://arxiv.org/abs/2308.07117) | iSTFTNet2 | 1D/2D multiband vocoder geometry |
| `hifigan/` | [2010.05646](https://arxiv.org/abs/2010.05646) | HiFi-GAN | Multi-period adversarial and feature matching losses |

Every folder contains the downloaded `paper.pdf` and its layout-preserving
`paper.txt` extraction. No checkpoints, datasets, or generated samples belong
in this tree.
