from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torch.utils.checkpoint import checkpoint

LRELU_SLOPE = 0.1


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _checkpoint(function, *args):
    return checkpoint(function, *args, use_reentrant=True)


def _magnitude_stft(
    waveform: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    spectrum = torch.stft(
        waveform,
        fft_size,
        hop_size,
        win_length,
        window.to(waveform.device),
        return_complex=True,
    )
    return spectrum.abs().transpose(2, 1)


class SpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        use_spectral_norm: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.out = norm(nn.Conv2d(32, 1, 3, padding=1))
        self.gradient_checkpointing = gradient_checkpointing
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _conv_block(self, features: torch.Tensor, index: int, dummy: torch.Tensor) -> torch.Tensor:
        del dummy
        return F.leaky_relu(self.convs[index](features), LRELU_SLOPE)

    def _out_block(self, features: torch.Tensor, dummy: torch.Tensor) -> torch.Tensor:
        del dummy
        return self.out(features)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = _magnitude_stft(
            waveform.squeeze(1),
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
        ).unsqueeze(1)
        maps: list[torch.Tensor] = []
        for index in range(len(self.convs)):
            if self.gradient_checkpointing:
                features = _checkpoint(self._conv_block, features, index, self.dummy)
            else:
                features = self._conv_block(features, index, self.dummy)
            maps.append(features)
        if self.gradient_checkpointing:
            features = _checkpoint(self._out_block, features, self.dummy)
        else:
            features = self._out_block(features, self.dummy)
        maps.append(features)
        return torch.flatten(features, 1), maps


class MultiResSpecDiscriminator(nn.Module):
    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        resolutions = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))
        self.discriminators = nn.ModuleList(
            SpecDiscriminator(fft, hop, window, gradient_checkpointing=gradient_checkpointing)
            for fft, hop, window in resolutions
        )

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real_logits, fake_logits, real_maps, fake_maps = [], [], [], []
        for discriminator in self.discriminators:
            real_logit, real_map = discriminator(real)
            fake_logit, fake_map = discriminator(fake)
            real_logits.append(real_logit)
            fake_logits.append(fake_logit)
            real_maps.append(real_map)
            fake_maps.append(fake_map)
        return real_logits, fake_logits, real_maps, fake_maps


class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.period = period
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                norm(nn.Conv2d(1024, 1024, (kernel_size, 1), padding=(2, 0))),
            ]
        )
        self.out = norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))
        self.gradient_checkpointing = gradient_checkpointing
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _conv_block(self, features: torch.Tensor, index: int, dummy: torch.Tensor) -> torch.Tensor:
        del dummy
        return F.leaky_relu(self.convs[index](features), LRELU_SLOPE)

    def _out_block(self, features: torch.Tensor, dummy: torch.Tensor) -> torch.Tensor:
        del dummy
        return self.out(features)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch, channels, samples = waveform.shape
        remainder = samples % self.period
        if remainder:
            waveform = F.pad(waveform, (0, self.period - remainder), "reflect")
            samples = waveform.shape[-1]
        features = waveform.view(batch, channels, samples // self.period, self.period)
        maps: list[torch.Tensor] = []
        for index in range(len(self.convs)):
            if self.gradient_checkpointing:
                features = _checkpoint(self._conv_block, features, index, self.dummy)
            else:
                features = self._conv_block(features, index, self.dummy)
            maps.append(features)
        if self.gradient_checkpointing:
            features = _checkpoint(self._out_block, features, self.dummy)
        else:
            features = self._out_block(features, self.dummy)
        maps.append(features)
        return torch.flatten(features, 1), maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(period, gradient_checkpointing=gradient_checkpointing)
            for period in (2, 3, 5, 7, 11)
        )

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real_logits, fake_logits, real_maps, fake_maps = [], [], [], []
        for discriminator in self.discriminators:
            real_logit, real_map = discriminator(real)
            fake_logit, fake_map = discriminator(fake)
            real_logits.append(real_logit)
            fake_logits.append(fake_logit)
            real_maps.append(real_map)
            fake_maps.append(fake_map)
        return real_logits, fake_logits, real_maps, fake_maps
