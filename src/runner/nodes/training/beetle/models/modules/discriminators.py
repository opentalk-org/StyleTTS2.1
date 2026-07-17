from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torch.utils.checkpoint import checkpoint


LEAKY_RELU_SLOPE = 0.1


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _checkpoint(function, *args):
    return checkpoint(function, *args, use_reentrant=True)


def _magnitude_stft(
    waveform: Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: Tensor,
) -> Tensor:
    spectrum = torch.stft(
        waveform,
        fft_size,
        hop_size,
        win_length,
        window.to(waveform.device),
        return_complex=True,
    )
    return spectrum.abs().transpose(2, 1)


class SpectrogramDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        use_spectral_norm: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        normalize = spectral_norm if use_spectral_norm else weight_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.convolutions = nn.ModuleList(
            [
                normalize(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                normalize(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.output = normalize(nn.Conv2d(32, 1, 3, padding=1))
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_anchor = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _convolution(self, features: Tensor, index: int, anchor: Tensor) -> Tensor:
        del anchor
        return F.leaky_relu(self.convolutions[index](features), LEAKY_RELU_SLOPE)

    def _output(self, features: Tensor, anchor: Tensor) -> Tensor:
        del anchor
        return self.output(features)

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        features = _magnitude_stft(
            waveform.squeeze(1),
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
        ).unsqueeze(1)
        feature_maps: list[Tensor] = []
        for index in range(len(self.convolutions)):
            if self.gradient_checkpointing:
                features = _checkpoint(
                    self._convolution,
                    features,
                    index,
                    self.checkpoint_anchor,
                )
            else:
                features = self._convolution(features, index, self.checkpoint_anchor)
            feature_maps.append(features)
        if self.gradient_checkpointing:
            features = _checkpoint(self._output, features, self.checkpoint_anchor)
        else:
            features = self._output(features, self.checkpoint_anchor)
        feature_maps.append(features)
        return torch.flatten(features, 1), feature_maps


class MultiResSpecDiscriminator(nn.Module):
    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        resolutions = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))
        self.discriminators = nn.ModuleList(
            SpectrogramDiscriminator(
                fft_size,
                hop_size,
                win_length,
                gradient_checkpointing=gradient_checkpointing,
            )
            for fft_size, hop_size, win_length in resolutions
        )

    def forward(self, real: Tensor, fake: Tensor):
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
        normalize = spectral_norm if use_spectral_norm else weight_norm
        self.period = period
        self.convolutions = nn.ModuleList(
            [
                normalize(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                normalize(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                normalize(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                normalize(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0))),
                normalize(nn.Conv2d(1024, 1024, (kernel_size, 1), padding=(2, 0))),
            ]
        )
        self.output = normalize(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_anchor = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _convolution(self, features: Tensor, index: int, anchor: Tensor) -> Tensor:
        del anchor
        return F.leaky_relu(self.convolutions[index](features), LEAKY_RELU_SLOPE)

    def _output(self, features: Tensor, anchor: Tensor) -> Tensor:
        del anchor
        return self.output(features)

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        batch, channels, samples = waveform.shape
        remainder = samples % self.period
        if remainder:
            waveform = F.pad(waveform, (0, self.period - remainder), "reflect")
            samples = waveform.shape[-1]
        features = waveform.view(batch, channels, samples // self.period, self.period)
        feature_maps: list[Tensor] = []
        for index in range(len(self.convolutions)):
            if self.gradient_checkpointing:
                features = _checkpoint(
                    self._convolution,
                    features,
                    index,
                    self.checkpoint_anchor,
                )
            else:
                features = self._convolution(features, index, self.checkpoint_anchor)
            feature_maps.append(features)
        if self.gradient_checkpointing:
            features = _checkpoint(self._output, features, self.checkpoint_anchor)
        else:
            features = self._output(features, self.checkpoint_anchor)
        feature_maps.append(features)
        return torch.flatten(features, 1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(period, gradient_checkpointing=gradient_checkpointing)
            for period in (2, 3, 5, 7, 11)
        )

    def forward(self, real: Tensor, fake: Tensor):
        real_logits, fake_logits, real_maps, fake_maps = [], [], [], []
        for discriminator in self.discriminators:
            real_logit, real_map = discriminator(real)
            fake_logit, fake_map = discriminator(fake)
            real_logits.append(real_logit)
            fake_logits.append(fake_logit)
            real_maps.append(real_map)
            fake_maps.append(fake_map)
        return real_logits, fake_logits, real_maps, fake_maps


@dataclass(frozen=True)
class DiscriminatorEvaluation:
    logits: list[Tensor]
    feature_maps: list[list[Tensor]]


@dataclass(frozen=True)
class StyleTTSDiscriminatorOutput:
    real: DiscriminatorEvaluation
    fake: DiscriminatorEvaluation


class StyleTTSDiscriminators(nn.Module):
    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.multi_period = MultiPeriodDiscriminator(gradient_checkpointing)
        self.multi_resolution = MultiResSpecDiscriminator(gradient_checkpointing)

    def forward(self, real: Tensor, fake: Tensor) -> StyleTTSDiscriminatorOutput:
        period = self.multi_period(real, fake)
        resolution = self.multi_resolution(real, fake)
        return StyleTTSDiscriminatorOutput(
            real=DiscriminatorEvaluation(
                logits=[*period[0], *resolution[0]],
                feature_maps=[*period[2], *resolution[2]],
            ),
            fake=DiscriminatorEvaluation(
                logits=[*period[1], *resolution[1]],
                feature_maps=[*period[3], *resolution[3]],
            ),
        )


def build_styletts_discriminators(
    gradient_checkpointing: bool = False,
) -> StyleTTSDiscriminators:
    return StyleTTSDiscriminators(gradient_checkpointing)
