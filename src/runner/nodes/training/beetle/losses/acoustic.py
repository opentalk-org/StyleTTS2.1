from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram


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


FREQUENCY_BANDS = (
    FrequencyBand(0, 1000),
    FrequencyBand(1000, 4000),
    FrequencyBand(4000, 8000),
    FrequencyBand(8000, 12000),
)


def _expanded_mask(values: Tensor, mask: Tensor) -> Tensor:
    return torch.broadcast_to(mask.to(dtype=torch.bool), values.shape)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = _expanded_mask(values, mask)
    count = expanded.sum()
    torch._assert_async(count > 0, "loss mask must select at least one element")
    return (values * expanded.to(dtype=values.dtype)).sum() / count


def masked_kl_standard_normal(mean: Tensor, log_scale: Tensor, mask: Tensor) -> Tensor:
    divergence = 0.5 * (mean.square() + torch.exp(2 * log_scale) - 1 - 2 * log_scale)
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


class StyleTTSMelTransform(nn.Module):
    def __init__(
        self,
        resolution: SpectralResolution,
        sample_rate: int,
        mel_channels: int,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=resolution.n_fft,
            win_length=resolution.win_length,
            hop_length=resolution.hop_length,
            window_fn=torch.hann_window,
            n_mels=mel_channels,
        )
        self.register_buffer(
            "stft_window",
            torch.hann_window(resolution.win_length),
            persistent=False,
        )
        mel_max = 2595 * torch.log10(torch.tensor(1 + sample_rate / 1400))
        mel_points = torch.linspace(0, mel_max, self.transform.n_mels + 2)[1:-1]
        centers = 700 * (torch.pow(10, mel_points / 2595) - 1)
        band_masks = torch.stack(
            tuple(
                (centers >= band.minimum_hz) & (centers < band.maximum_hz)
                for band in FREQUENCY_BANDS
            )
        )
        self.register_buffer("frequency_band_masks", band_masks, persistent=False)

    def forward(self, waveform: Tensor) -> Tensor:
        mel = self.transform(waveform)
        return (torch.log(1e-5 + mel) + 4) / 4

    def complex_spectrum(self, waveform: Tensor) -> Tensor:
        return torch.stft(
            waveform,
            n_fft=self.resolution.n_fft,
            hop_length=self.resolution.hop_length,
            win_length=self.resolution.win_length,
            window=self.stft_window,
            return_complex=True,
        )


class MultiResolutionReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        mel_channels: int,
        complex_reconstruction_steps: int,
        resolutions: tuple[SpectralResolution, ...] = (
            SpectralResolution(1024, 120, 600),
            SpectralResolution(2048, 240, 1200),
            SpectralResolution(512, 50, 240),
        ),
    ) -> None:
        super().__init__()
        self.complex_reconstruction_steps = complex_reconstruction_steps
        self.transforms = nn.ModuleList(
            StyleTTSMelTransform(resolution, sample_rate, mel_channels)
            for resolution in resolutions
        )

    def _resolution_loss(
        self,
        transform: StyleTTSMelTransform,
        predicted: Tensor,
        target: Tensor,
        lengths: Tensor,
        include_diagnostics: bool,
        include_complex: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        total = predicted.new_zeros(())
        band_totals = (
            tuple(predicted.new_zeros(()) for _ in FREQUENCY_BANDS)
            if include_diagnostics
            else ()
        )
        for length in torch.unique(lengths):
            selected = lengths == length
            sample_count = selected.sum()
            valid_samples = int(length.item())
            predicted_mel = transform(predicted[selected, 0, :valid_samples])
            target_mel = transform(target[selected, 0, :valid_samples])
            difference = (target_mel - predicted_mel).abs()
            convergence = difference.sum()
            convergence = convergence / torch.norm(target_mel, p=1)
            if include_complex:
                predicted_spectrum = transform.complex_spectrum(
                    predicted[selected, 0, :valid_samples]
                )
                target_spectrum = transform.complex_spectrum(
                    target[selected, 0, :valid_samples]
                )
                complex_convergence = (
                    target_spectrum - predicted_spectrum
                ).abs().sum()
                complex_convergence = (
                    complex_convergence / target_spectrum.abs().sum()
                )
                convergence = (convergence + complex_convergence) * 0.5
            total = total + convergence * sample_count / predicted.shape[0]
            if include_diagnostics:
                band_totals = tuple(
                    band_total
                    + self._band_loss(transform, difference, target_mel, band_index)
                    * sample_count
                    / predicted.shape[0]
                    for band_index, band_total in enumerate(
                        band_totals,
                    )
                )
        return total, band_totals

    @staticmethod
    def _band_loss(
        transform: StyleTTSMelTransform,
        difference: Tensor,
        target: Tensor,
        band_index: int,
    ) -> Tensor:
        selected = transform.frequency_band_masks[band_index]
        denominator = target[:, selected, :].abs().sum()
        return difference[:, selected, :].sum() / denominator

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        sample_mask: Tensor,
        completed_step: int,
        include_diagnostics: bool = False,
    ) -> ReconstructionLoss:
        lengths = sample_mask.sum(dim=(1, 2))
        torch._assert_async(
            torch.all(lengths > 0),
            "reconstruction loss requires valid waveform samples",
        )
        results = tuple(
            self._resolution_loss(
                transform,
                predicted,
                target,
                lengths,
                include_diagnostics,
                completed_step <= self.complex_reconstruction_steps,
            )
            for transform in self.transforms
        )
        losses = torch.stack(tuple(result[0] for result in results))
        mel = losses.mean()
        band_losses = (
            torch.stack(tuple(torch.stack(result[1]) for result in results)).mean(dim=0)
            if include_diagnostics
            else ()
        )
        return ReconstructionLoss(
            mel=mel,
            total=mel,
            resolutions=tuple(
                ResolutionLoss(transform.resolution, value)
                for transform, value in zip(self.transforms, losses, strict=True)
            ),
            bands=(
                tuple(
                    FrequencyBandLoss(band, value)
                    for band, value in zip(
                        FREQUENCY_BANDS,
                        band_losses,
                        strict=True,
                    )
                )
                if include_diagnostics
                else ()
            ),
        )
