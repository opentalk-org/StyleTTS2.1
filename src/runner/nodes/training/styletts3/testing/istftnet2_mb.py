"""Standalone native-hop-300 iSTFTNet2-MB vocoder scaffold."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from runner.nodes.training.styletts3.istftnet2_mb import (
    ISTFTNet2MBCore,
)


SAMPLE_RATE = 24_000


class ISTFTNet2MB(nn.Module):
    def __init__(
        self,
        mel_channels: int = 80,
        base_channels: int = 128,
        bands: int = 4,
        nfft: int = 60,
    ) -> None:
        super().__init__()
        assert base_channels == 128, "the shared iSTFTNet2-MB core uses 128 frame channels"
        assert bands == 4, "iSTFTNet2-MB uses four PQMF subbands"
        assert nfft == 60, "native-hop-300 C5-I15-B4 synthesis uses a 60-point iSTFT"
        self.mel_channels = mel_channels
        self.conv_pre = nn.Conv1d(mel_channels, base_channels, 7, padding=3)
        self.core = ISTFTNet2MBCore()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        assert mel.ndim == 3, f"expected rank-3 mel conditioning, got rank {mel.ndim}"
        assert mel.shape[1] == self.mel_channels, (
            f"expected {self.mel_channels} mel channels, got {mel.shape[1]}"
        )
        return self.core(self.conv_pre(mel))


def _flops(module: nn.Module, conditioning: torch.Tensor) -> int:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        module(conditioning)
    return counter.get_total_flops()


def benchmark(frames: int = 300) -> None:
    torch.manual_seed(0)
    conditioning = torch.randn(1, 80, frames)
    generator = ISTFTNet2MB().eval()
    parameters = sum(parameter.numel() for parameter in generator.parameters())
    waveform = generator(conditioning)
    flops = _flops(generator, conditioning)
    print(
        f"iSTFTNet2-MB params={parameters / 1e6:.3f} M "
        f"output={waveform.shape[-1]} ({waveform.shape[-1] / SAMPLE_RATE:.3f} s)"
    )
    print(f"GFLOP={flops / 1e9:.3f} GMAC={flops / 2e9:.3f}")


if __name__ == "__main__":
    benchmark()

