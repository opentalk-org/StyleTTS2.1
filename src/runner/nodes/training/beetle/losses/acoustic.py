from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram


@dataclass(frozen=True)
class SpectralResolution:
    n_fft: int
    hop_length: int
    win_length: int


@dataclass(frozen=True)
class ReconstructionLoss:
    mel: Tensor
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
    divergence = 0.5 * (mean.square() + torch.exp(2 * log_scale) - 1 - 2 * log_scale)
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
    if (
        not predictions
        or len(predictions) != len(targets)
        or len(targets) != len(masks)
    ):
        raise ValueError("reconstruction views, targets, and masks must align")
    losses = [
        _masked_mean((prediction - target).abs(), mask)
        for prediction, target, mask in zip(predictions, targets, masks, strict=True)
    ]
    return torch.stack(losses).mean()


class StyleTTSMelTransform(nn.Module):
    def __init__(
        self,
        resolution: SpectralResolution,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=resolution.n_fft,
            win_length=resolution.win_length,
            hop_length=resolution.hop_length,
            window_fn=torch.hann_window,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        mel = self.transform(waveform)
        return (torch.log(1e-5 + mel) + 4) / 4


class MultiResolutionReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        resolutions: tuple[SpectralResolution, ...] = (
            SpectralResolution(1024, 120, 600),
            SpectralResolution(2048, 240, 1200),
            SpectralResolution(512, 50, 240),
        ),
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(
            StyleTTSMelTransform(resolution, sample_rate) for resolution in resolutions
        )

    def _resolution_loss(
        self,
        transform: StyleTTSMelTransform,
        predicted: Tensor,
        target: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        total = predicted.new_zeros(())
        for length in torch.unique(lengths):
            selected = lengths == length
            sample_count = selected.sum()
            valid_samples = int(length.item())
            predicted_mel = transform(predicted[selected, 0, :valid_samples])
            target_mel = transform(target[selected, 0, :valid_samples])
            convergence = torch.norm(target_mel - predicted_mel, p=1)
            convergence = convergence / torch.norm(target_mel, p=1)
            total = total + convergence * sample_count / predicted.shape[0]
        return total

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        sample_mask: Tensor,
    ) -> ReconstructionLoss:
        if predicted.shape != target.shape or predicted.shape != sample_mask.shape:
            raise ValueError("waveforms and sample mask must have equal shapes")
        lengths = sample_mask.sum(dim=(1, 2))
        if torch.any(lengths == 0):
            raise ValueError("reconstruction loss requires valid waveform samples")
        losses = torch.stack(
            tuple(
                self._resolution_loss(transform, predicted, target, lengths)
                for transform in self.transforms
            )
        )
        mel = losses.mean()
        return ReconstructionLoss(mel=mel, total=mel)
