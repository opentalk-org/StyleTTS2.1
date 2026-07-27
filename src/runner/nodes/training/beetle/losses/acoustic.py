from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.functional import melscale_fbanks


@dataclass(frozen=True)
class SpectralResolution:
    n_fft: int
    hop_length: int
    win_length: int


@dataclass(frozen=True)
class FrequencyBand:
    minimum_hz: int
    maximum_hz: int


@dataclass(frozen=True)
class ResolutionLoss:
    resolution: SpectralResolution
    value: Tensor


@dataclass(frozen=True)
class FrequencyBandLoss:
    band: FrequencyBand
    value: Tensor


@dataclass(frozen=True)
class ReconstructionLoss:
    mel: Tensor
    total: Tensor
    resolutions: tuple[ResolutionLoss, ...]
    bands: tuple[FrequencyBandLoss, ...]


def _expanded_mask(values: Tensor, mask: Tensor) -> Tensor:
    return torch.broadcast_to(mask.to(dtype=torch.bool), values.shape)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = _expanded_mask(values, mask)
    count = expanded.sum()
    torch._assert_async(count > 0, "loss mask must select at least one element")
    return (values * expanded.to(dtype=values.dtype)).sum() / count


def masked_kl_standard_normal(mean: Tensor, log_scale: Tensor, mask: Tensor) -> Tensor:
    divergence = 0.5 * (
        mean.square() + torch.exp(2 * log_scale) - 1 - 2 * log_scale
    )
    return _masked_mean(divergence.sum(dim=1, keepdim=True), mask)


def masked_f0_smooth_l1(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    scale_hz: float,
) -> Tensor:
    values = F.smooth_l1_loss(
        predicted / scale_hz,
        target / scale_hz,
        reduction="none",
    )
    return _masked_mean(values, mask[:, 0])


def masked_n_smooth_l1(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    values = F.smooth_l1_loss(predicted, target, reduction="none")
    return _masked_mean(values, mask[:, 0])


def multiresolution_l1(
    predictions: tuple[Tensor, ...],
    targets: tuple[Tensor, ...],
    masks: tuple[Tensor, ...],
) -> Tensor:
    losses = [
        _masked_mean((prediction - target).abs(), mask)
        for prediction, target, mask in zip(predictions, targets, masks, strict=True)
    ]
    return torch.stack(losses).mean()


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        mel_channels: int,
        f_min: float,
        f_max: float,
    ) -> None:
        super().__init__()
        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=mel_channels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        padding = (self.n_fft - self.hop_length) // 2
        padded = F.pad(
            waveform.unsqueeze(1),
            (padding, padding),
            mode="reflect",
        ).squeeze(1)
        spectrum = torch.stft(
            padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitude = torch.sqrt(
            torch.view_as_real(spectrum).square().sum(dim=-1) + 1e-9
        )
        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))


class HiFTNetReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        mel_channels: int,
        f_min: float,
        f_max: float,
    ) -> None:
        super().__init__()
        self.resolution = SpectralResolution(n_fft, hop_length, win_length)
        self.transforms = nn.ModuleList(
            (
                LogMelSpectrogram(
                    sample_rate,
                    n_fft,
                    hop_length,
                    win_length,
                    mel_channels,
                    f_min,
                    f_max,
                ),
            )
        )

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        sample_mask: Tensor,
        completed_step: int,
        include_diagnostics: bool = False,
    ) -> ReconstructionLoss:
        del sample_mask, completed_step, include_diagnostics
        predicted_mel = self.transforms[0](predicted[:, 0])
        target_mel = self.transforms[0](target[:, 0])
        mel = F.l1_loss(predicted_mel, target_mel)
        return ReconstructionLoss(
            mel,
            mel,
            (ResolutionLoss(self.resolution, mel),),
            (),
        )
