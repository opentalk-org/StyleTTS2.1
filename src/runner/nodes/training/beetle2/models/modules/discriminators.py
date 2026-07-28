from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


LEAKY_RELU_SLOPE = 0.1


class SpectrogramDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size: int,
        shift_size: int,
        win_length: int,
    ) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.convolutions = nn.ModuleList(
            (
                weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=1)),
            )
        )
        self.output = weight_norm(nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        spectrum = torch.stft(
            waveform.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.shift_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        features = spectrum.abs().transpose(1, 2).unsqueeze(1)
        feature_maps = []
        for convolution in self.convolutions:
            features = F.leaky_relu(convolution(features), LEAKY_RELU_SLOPE)
            feature_maps.append(features)
        features = self.output(features)
        feature_maps.append(features)
        return torch.flatten(features, 1), feature_maps


class MultiResSpecDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        resolutions = (
            (1024, 120, 600),
            (2048, 240, 1200),
            (512, 50, 240),
        )
        self.discriminators = nn.ModuleList(
            SpectrogramDiscriminator(fft_size, shift_size, win_length)
            for fft_size, shift_size, win_length in resolutions
        )

    def forward(
        self,
        real: Tensor,
        fake: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
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
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self.convolutions = nn.ModuleList(
            (
                weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (5, 1), padding=(2, 0))),
            )
        )
        self.output = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        batch, channels, samples = waveform.shape
        remainder = samples % self.period
        if remainder:
            waveform = F.pad(
                waveform,
                (0, self.period - remainder),
                mode="reflect",
            )
            samples = waveform.shape[-1]
        features = waveform.view(
            batch,
            channels,
            samples // self.period,
            self.period,
        )
        feature_maps = []
        for convolution in self.convolutions:
            features = F.leaky_relu(convolution(features), LEAKY_RELU_SLOPE)
            feature_maps.append(features)
        features = self.output(features)
        feature_maps.append(features)
        return torch.flatten(features, 1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(period) for period in (2, 3, 5, 7, 11)
        )

    def forward(
        self,
        real: Tensor,
        fake: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
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
    period_count: int


class StyleTTSDiscriminators(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multi_period = MultiPeriodDiscriminator()
        self.multi_resolution = MultiResSpecDiscriminator()

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
            period_count=len(period[0]),
        )


def build_styletts_discriminators() -> StyleTTSDiscriminators:
    return StyleTTSDiscriminators()
