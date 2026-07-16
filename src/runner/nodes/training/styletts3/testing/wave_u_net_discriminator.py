"""Wave-U-Net Discriminator (Kaneko et al., 2023 — arXiv:2303.13909).

Faithful single-file reimplementation of the discriminator described in
"Wave-U-Net Discriminator: Fast and Lightweight Discriminator for
Generative Adversarial Network-Based Speech Synthesis".

The network is a single, sample-wise discriminator with a Wave-U-Net
(encoder-decoder + skip connections) shape. It emits one real/fake logit
per input sample and exposes every block output as a feature map for the
feature-matching loss.

Every architectural detail is transcribed from Figures 2 and 3:

  Fig. 2 (overall):
    input waveform (B, 1, T)
      ResBlockDown c32,  s3
      ResBlockDown c64,  s3
      ResBlockDown c128, s3
      ResBlockDown c256, s3
      ResBlockDown c512, s3          <- bottleneck
      ResBlockUp   c256, s3
      Concat(with ResBlockDown c256 out)
      ResBlockUp   c128, s3
      Concat(with ResBlockDown c128 out)
      ResBlockUp   c64,  s3
      Concat(with ResBlockDown c64  out)
      ResBlockUp   c32,  s3
      Concat(with ResBlockDown c32  out)
      ResBlockUp   c32,  s3
      Conv c1, k5, s1                -> sample-wise logits (B, 1, T)

  Fig. 3 (residual blocks), quirks:
    - LReLU with negative slope 0.1 (pre-activation: LReLU -> Conv).
    - Global Norm at the end of every block:
        b = a / sqrt( (1/N) * sum_i (a_i)^2 + eps ),  eps = 1e-8,
      where N is the total number of features (channels x time) per sample;
      no trainable parameters.
    - Residual-path output scaled by a constant 0.4 before the block-level add.
    - ResBlockDown:
        skip     = Concat( Conv1x1(AvgPool_s3(x)), AvgPool_s3(x) )
        residual = LReLU(x) -> [ Conv k6 s3  +  Dup-channels(AvgPool_s3(.)) ]
                            -> LReLU -> Conv k5 s1
        out      = GlobalNorm( skip + 0.4 * residual )
      "Dup. channels" tiles the pooled features up to the output channel count.
    - ResBlockUp:
        skip     = Upsample_s3( Conv1x1(x) )
        residual = LReLU(x) -> [ ConvT k6 s3  +  Drop-channels(Upsample_s3(.)) ]
                            -> LReLU -> Conv k5 s1
        out      = GlobalNorm( skip + 0.4 * residual )
      "Drop. channels" keeps the leading output-channel slice of the upsampled
      features (the inverse of Dup-channels).

The strided (s3) resampling makes intermediate lengths drift by a few samples;
branches are cropped to their common length before every add/concat, and the
final logits are matched to the input length. This preserves the sample-wise
property without inventing padding the paper does not specify.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 3
LRELU_SLOPE = 0.1
RESIDUAL_SCALE = 0.4
GLOBAL_NORM_EPS = 1e-8
ENCODER_CHANNELS = (32, 64, 128, 256, 512)


def _crop_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop two (B, C, T) tensors to their shared time length before combining."""
    t = min(a.shape[-1], b.shape[-1])
    return a[..., :t], b[..., :t]


class GlobalNorm(nn.Module):
    """Parameter-free RMS normalization over all features of a sample (Eq. 4).

    Regularizes the norm of the whole feature map so the deep encoder-decoder
    discriminator cannot collapse onto a few features. N spans channels x time.
    """

    def __init__(self, eps: float = GLOBAL_NORM_EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=(1, 2), keepdim=True) + self.eps)
        return x / rms


class ResBlockDown(nn.Module):
    """Residual downsampling block (Fig. 3a). Channels grow c_in -> c_out, /3 time."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        assert c_out % c_in == 0, "Dup-channels needs c_out divisible by c_in"
        self.c_in, self.c_out = c_in, c_out

        # Skip: pool, then concat the pooled input with a 1x1 projection of it.
        self.skip_pool = nn.AvgPool1d(kernel_size=STRIDE, stride=STRIDE)
        self.skip_conv = nn.Conv1d(c_in, c_out - c_in, kernel_size=1)

        # Residual: strided conv (downsample + channel change) added to a pooled,
        # channel-duplicated copy, then LReLU and a same-length conv.
        self.res_conv_down = nn.Conv1d(c_in, c_out, kernel_size=6, stride=STRIDE, padding=2)
        self.res_pool = nn.AvgPool1d(kernel_size=STRIDE, stride=STRIDE)
        self.res_conv_out = nn.Conv1d(c_out, c_out, kernel_size=5, stride=1, padding=2)

        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.norm = GlobalNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.skip_pool(x)
        skip = torch.cat([self.skip_conv(pooled), pooled], dim=1)

        r = self.act(x)
        conv_branch = self.res_conv_down(r)
        dup_branch = self.res_pool(r).repeat(1, self.c_out // self.c_in, 1)
        conv_branch, dup_branch = _crop_pair(conv_branch, dup_branch)
        r = self.act(conv_branch + dup_branch)
        residual = self.res_conv_out(r)

        skip, residual = _crop_pair(skip, residual)
        return self.norm(skip + RESIDUAL_SCALE * residual)


class ResBlockUp(nn.Module):
    """Residual upsampling block (Fig. 3b). Channels shrink c_in -> c_out, x3 time."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        assert c_in % c_out == 0, "Drop-channels needs c_in divisible by c_out"
        self.c_in, self.c_out = c_in, c_out

        # Skip: 1x1 projection then nearest upsample.
        self.skip_conv = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.skip_up = nn.Upsample(scale_factor=STRIDE, mode="nearest")

        # Residual: transposed conv (upsample + channel change) added to an
        # upsampled, channel-dropped copy, then LReLU and a same-length conv.
        self.res_convt_up = nn.ConvTranspose1d(c_in, c_out, kernel_size=6, stride=STRIDE)
        self.res_up = nn.Upsample(scale_factor=STRIDE, mode="nearest")
        self.res_conv_out = nn.Conv1d(c_out, c_out, kernel_size=5, stride=1, padding=2)

        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.norm = GlobalNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_up(self.skip_conv(x))

        r = self.act(x)
        convt_branch = self.res_convt_up(r)
        drop_branch = self.res_up(r)[:, : self.c_out]
        convt_branch, drop_branch = _crop_pair(convt_branch, drop_branch)
        r = self.act(convt_branch + drop_branch)
        residual = self.res_conv_out(r)

        skip, residual = _crop_pair(skip, residual)
        return self.norm(skip + RESIDUAL_SCALE * residual)


class WaveUNetDiscriminator(nn.Module):
    """Single sample-wise Wave-U-Net discriminator (arXiv:2303.13909, Fig. 2).

    forward returns (logits, features):
      - logits:   (B, 1, T) per-sample real/fake scores.
      - features: list of every block output, used by the feature-matching loss.
    """

    def __init__(self) -> None:
        super().__init__()
        down_in = (1,) + ENCODER_CHANNELS[:-1]
        self.downs = nn.ModuleList(
            ResBlockDown(c_in, c_out) for c_in, c_out in zip(down_in, ENCODER_CHANNELS)
        )

        # Decoder mirrors the encoder; every stage after the first receives the
        # concatenated encoder skip, doubling its input channels.
        up_out = (256, 128, 64, 32, 32)
        up_in = (512, 512, 256, 128, 64)
        self.ups = nn.ModuleList(
            ResBlockUp(c_in, c_out) for c_in, c_out in zip(up_in, up_out)
        )
        self.out_conv = nn.Conv1d(up_out[-1], 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        length = x.shape[-1]
        features: list[torch.Tensor] = []

        skips: list[torch.Tensor] = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)
            features.append(h)

        h = skips[-1]
        for i, up in enumerate(self.ups):
            if i > 0:
                skip = skips[len(skips) - 1 - i]
                h, skip = _crop_pair(h, skip)
                h = torch.cat([h, skip], dim=1)
            h = up(h)
            features.append(h)

        logits = self.out_conv(h)
        if logits.shape[-1] >= length:
            logits = logits[..., :length]
        else:
            logits = F.pad(logits, (0, length - logits.shape[-1]))
        features.append(logits)
        return logits, features


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator objective (Eq. 1): (D(x)-1)^2 + D(G(s))^2."""
    return (real_logits - 1.0).pow(2).mean() + fake_logits.pow(2).mean()


def generator_adv_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """LSGAN generator objective (Eq. 2): (D(G(s))-1)^2."""
    return (fake_logits - 1.0).pow(2).mean()


def feature_matching_loss(
    real_features: list[torch.Tensor], fake_features: list[torch.Tensor]
) -> torch.Tensor:
    """L1 feature-matching loss (Eq. 3), averaged per layer by feature count."""
    total = real_features[0].new_zeros(())
    for real, fake in zip(real_features, fake_features):
        real, fake = _crop_pair(real, fake)
        total = total + (real - fake).abs().mean()
    return total


if __name__ == "__main__":
    torch.manual_seed(0)
    disc = WaveUNetDiscriminator()
    n_params = sum(p.numel() for p in disc.parameters())
    print(f"parameters: {n_params/1e6:.3f} M")

    # Paper trains on 8192-sample segments at 22.05 kHz.
    wav = torch.randn(2, 1, 8192)
    logits, feats = disc(wav)
    print(f"input : {tuple(wav.shape)}")
    print(f"logits: {tuple(logits.shape)}  (sample-wise, matches input length)")
    print(f"feature maps: {len(feats)}")
    for i, f in enumerate(feats):
        print(f"  layer {i:2d}: {tuple(f.shape)}")

    fake = torch.randn(2, 1, 8192)
    real_logits, real_feats = disc(wav)
    fake_logits, fake_feats = disc(fake)
    print(f"D loss : {discriminator_loss(real_logits, fake_logits).item():.4f}")
    print(f"G adv  : {generator_adv_loss(fake_logits).item():.4f}")
    print(f"FM loss: {feature_matching_loss(real_feats, fake_feats).item():.4f}")

    # Non-8192 length still yields sample-wise logits.
    odd = torch.randn(1, 1, 16000)
    odd_logits, _ = disc(odd)
    assert odd_logits.shape[-1] == 16000
    print(f"arbitrary length {16000} -> logits {tuple(odd_logits.shape)}  OK")
