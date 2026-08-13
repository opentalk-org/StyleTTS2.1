# StyleTTS-ZS Paper Mismatches

Reference: [StyleTTS-ZS, arXiv:2409.10058](https://arxiv.org/pdf/2409.10058)

## Resolved mismatches

### Duration GAN uses full phoneme lengths

Status: resolved

The duration discriminator operates on phoneme-length sequences. Both its
discriminator update and generator/feature-matching update now use the full
`input_lengths`. Divided lengths remain only where tensors genuinely have
half-rate mel/prosody resolution because of pooling or aligner downsampling.

