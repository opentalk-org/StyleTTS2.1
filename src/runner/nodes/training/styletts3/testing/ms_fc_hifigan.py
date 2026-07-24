"""MS-FC-HiFi-GAN from Yamashita et al., IEEE Access 2024."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

LRELU_SLOPE = 0.1


def pad_same(kernel: int, dilation: int = 1) -> int:
    return (kernel * dilation - dilation) // 2


class ResBlock(nn.Module):
    """HiFi-GAN V1 multi-receptive-field residual block."""

    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel, 1, dilation=d, padding=pad_same(kernel, d)) for d in dilations]
        )
        self.convs2 = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel, 1, dilation=1, padding=pad_same(kernel)) for _ in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = conv2(F.leaky_relu(conv1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
            x = x + xt
        return x


class MSFCHiFiGAN(nn.Module):
    def __init__(
        self,
        mel_channels: int = 80,
        init_channel: int = 512,
        upsample_rates: tuple[int, ...] = (4, 4),
        upsample_kernels: tuple[int, ...] = (8, 8),
        resblock_kernels: tuple[int, ...] = (3, 7, 11),
        subbands: int = 4,
        fc_m: int = 16,
        fc_n: int = 4,
        pqmf_taps: int = 62,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernels)
        self.subbands = subbands
        self.fc_n = fc_n
        self.conv_pre = nn.Conv1d(mel_channels, init_channel, 7, 1, 3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = init_channel
        for rate, kernel in zip(upsample_rates, upsample_kernels):
            self.ups.append(nn.ConvTranspose1d(channels, channels // 2, kernel, rate, padding=(kernel - rate) // 2))
            channels //= 2
            for rkernel in resblock_kernels:
                self.resblocks.append(ResBlock(channels, rkernel))
        self.conv_post = nn.Conv1d(channels, subbands * (fc_m + 2), 7, 1, 3)
        self.fc = nn.Conv1d(subbands * (fc_m + 2), subbands * fc_n, 1, groups=subbands, bias=False)
        self.register_buffer("pqmf", torch.randn(subbands, 1, pqmf_taps))
        self.pqmf_pad = (pqmf_taps - subbands) // 2

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)
        for stage, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            summed = None
            for kernel_index in range(self.num_kernels):
                residual = self.resblocks[stage * self.num_kernels + kernel_index](x)
                summed = residual if summed is None else summed + residual
            x = summed / self.num_kernels
        x = self.conv_post(F.leaky_relu(x, LRELU_SLOPE))
        x = self.fc(x)
        batch, _, frames = x.shape
        x = x.view(batch, self.subbands, self.fc_n, frames).reshape(batch, self.subbands, self.fc_n * frames)
        return F.conv_transpose1d(x, self.pqmf, stride=self.subbands, padding=self.pqmf_pad)


def benchmark(frames: int = 300) -> None:
    """Forward a T=frames mel and report FLOPs. frames=300 -> 3.2 s @ 24 kHz."""
    torch.manual_seed(0)
    generator = MSFCHiFiGAN().eval()
    params = sum(p.numel() for p in generator.parameters())
    mel = torch.randn(1, 80, frames)
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            audio = generator(mel)
    total = counter.get_total_flops()
    print(f"params: {params / 1e6:.2f} M")
    print(f"output samples: {audio.shape[-1]} (~{audio.shape[-1] / 24000:.3f} s)")
    print(f"GFLOP (2*MAC): {total / 1e9:.3f}   GMAC: {total / 2e9:.3f}")


if __name__ == "__main__":
    benchmark()
