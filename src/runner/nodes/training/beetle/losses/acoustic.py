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
class ReconstructionLoss:
    mel: Tensor
    spectrum: Tensor
    total: Tensor


def _expanded_mask(values: Tensor, mask: Tensor) -> Tensor:
    if mask.ndim != values.ndim:
        raise ValueError("loss mask rank must match values")
    try:
        return torch.broadcast_to(mask.to(dtype=torch.bool), values.shape)
    except RuntimeError as error:
        raise ValueError("loss mask must broadcast to values") from error


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = _expanded_mask(values, mask)
    count = expanded.sum()
    if count == 0:
        raise ValueError("loss mask must select at least one element")
    return (values * expanded.to(dtype=values.dtype)).sum() / count


def masked_kl_standard_normal(mean: Tensor, log_scale: Tensor, mask: Tensor) -> Tensor:
    if mean.shape != log_scale.shape:
        raise ValueError("posterior mean and log scale must have equal shapes")
    divergence = 0.5 * (
        mean.square() + torch.exp(2 * log_scale) - 1 - 2 * log_scale
    )
    return _masked_mean(divergence, mask)


def masked_f0_mse(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target F0 must have equal shapes")
    voiced_mask = mask[:, 0].to(dtype=torch.bool) & (target > 0)
    return _masked_mean((predicted - target).square(), voiced_mask)


def masked_n_mse(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target N must have equal shapes")
    return _masked_mean((predicted - target).square(), mask[:, 0])


def multiresolution_l1(
    predictions: tuple[Tensor, ...],
    targets: tuple[Tensor, ...],
    masks: tuple[Tensor, ...],
) -> Tensor:
    if not predictions or len(predictions) != len(targets) or len(targets) != len(masks):
        raise ValueError("reconstruction views, targets, and masks must align")
    losses = [
        _masked_mean((prediction - target).abs(), mask)
        for prediction, target, mask in zip(predictions, targets, masks, strict=True)
    ]
    return torch.stack(losses).mean()


class SpectralTransform(nn.Module):
    def __init__(
        self,
        resolution: SpectralResolution,
        sample_rate: int,
        mel_channels: int,
        f_min: float,
        f_max: float,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.register_buffer("window", torch.hann_window(resolution.win_length))
        mel_filter = melscale_fbanks(
            resolution.n_fft // 2 + 1,
            f_min,
            f_max,
            mel_channels,
            sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1)
        self.register_buffer("mel_filter", mel_filter)

    def forward(self, waveform: Tensor, sample_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        resolution = self.resolution
        spectrum = torch.stft(
            waveform[:, 0] * sample_mask[:, 0].to(dtype=waveform.dtype),
            n_fft=resolution.n_fft,
            hop_length=resolution.hop_length,
            win_length=resolution.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.log1p(spectrum.abs())
        mel = torch.matmul(self.mel_filter.to(dtype=magnitude.dtype), magnitude)
        valid_fraction = F.avg_pool1d(
            sample_mask.to(dtype=waveform.dtype),
            kernel_size=resolution.n_fft,
            stride=resolution.hop_length,
            padding=resolution.n_fft // 2,
            count_include_pad=True,
        )
        frame_mask = valid_fraction == 1
        if frame_mask.shape[-1] != magnitude.shape[-1]:
            raise ValueError("spectral mask does not match STFT frames")
        return magnitude, mel, frame_mask


class MultiResolutionReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        mel_channels: int,
        f_min: float,
        f_max: float,
        resolutions: tuple[SpectralResolution, ...] = (
            SpectralResolution(1024, 120, 600),
            SpectralResolution(2048, 240, 1200),
            SpectralResolution(512, 50, 240),
        ),
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(
            SpectralTransform(
                resolution,
                sample_rate,
                mel_channels,
                f_min,
                f_max,
            )
            for resolution in resolutions
        )

    def forward(self, predicted: Tensor, target: Tensor, sample_mask: Tensor) -> ReconstructionLoss:
        if predicted.shape != target.shape or predicted.shape != sample_mask.shape:
            raise ValueError("waveforms and sample mask must have equal shapes")
        predicted_views = [transform(predicted, sample_mask) for transform in self.transforms]
        target_views = [transform(target, sample_mask) for transform in self.transforms]
        frame_masks = tuple(view[2] for view in predicted_views)
        spectrum = multiresolution_l1(
            tuple(view[0] for view in predicted_views),
            tuple(view[0] for view in target_views),
            frame_masks,
        )
        mel = multiresolution_l1(
            tuple(view[1] for view in predicted_views),
            tuple(view[1] for view in target_views),
            frame_masks,
        )
        return ReconstructionLoss(mel=mel, spectrum=spectrum, total=mel + spectrum)
