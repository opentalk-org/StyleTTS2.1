from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram

from ..config import BeetleConfig
from ..data.records import BeetleBatch
from ..models.model import normalized_log_mel_energy
from ..models.modules.audio import AcousticFeatures
from ..models.modules.decoder import DecoderOutput
from ..models.modules.embeddings import AcousticStatistics
from ..models.stage2 import Stage2Models


@dataclass(frozen=True)
class MelBatch:
    values: Tensor
    mask: Tensor


@dataclass(frozen=True)
class ConditionalSynthesis:
    acoustic: AcousticFeatures
    decoded: DecoderOutput
    waveform: Tensor
    sample_mask: Tensor


class WaveformMelExtractor(nn.Module):
    def __init__(self, config: BeetleConfig) -> None:
        super().__init__()
        audio = config.audio
        self.hop_length = audio.hop_length
        self.n_fft = audio.n_fft
        self.transform = MelSpectrogram(
            sample_rate=audio.sample_rate,
            n_fft=audio.n_fft,
            win_length=audio.win_length,
            hop_length=audio.hop_length,
            f_min=audio.f_min,
            f_max=audio.f_max,
            n_mels=audio.mel_channels,
            power=1.0,
            center=True,
            normalized=False,
        )

    def forward(self, waveform: Tensor, lengths: Tensor) -> MelBatch:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("context waveform must have shape [B,1,S]")
        if lengths.shape != (waveform.shape[0],):
            raise ValueError("context waveform lengths must have shape [B]")
        required = max(waveform.shape[2], self.n_fft)
        padded = torch.nn.functional.pad(waveform, (0, required - waveform.shape[2]))
        mel = torch.log(self.transform(padded[:, 0]).clamp_min(1e-5))
        frame_lengths = torch.div(lengths, self.hop_length, rounding_mode="floor") + 1
        maximum = int(frame_lengths.max().clamp_min(1))
        maximum += maximum % 2
        mel = mel[:, :, :maximum]
        if mel.shape[2] < maximum:
            mel = torch.nn.functional.pad(mel, (0, maximum - mel.shape[2]))
        positions = torch.arange(maximum, device=waveform.device).unsqueeze(0)
        mask = (positions < frame_lengths.unsqueeze(1)).unsqueeze(1)
        return MelBatch(mel, mask)


def expand_vector(values: Tensor, frames: int) -> Tensor:
    return values.unsqueeze(2).expand(-1, -1, frames)


def boundary_pool(
    tokens: Tensor,
    mask: Tensor,
    available: Tensor,
    counts: Tensor,
    pre: bool,
) -> Tensor:
    lengths = mask.sum(dim=1)
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    selected = (
        positions >= (lengths - counts).clamp_min(0).unsqueeze(1)
        if pre
        else positions < counts.unsqueeze(1)
    )
    selected = selected & mask & available.unsqueeze(1)
    numeric = selected.unsqueeze(1).to(dtype=tokens.dtype)
    return (tokens * numeric).sum(dim=2) / numeric.sum(dim=2).clamp_min(1)


def group_ids(views: Tensor, device: torch.device) -> Tensor:
    groups, view_count = views.shape[:2]
    return torch.arange(groups, device=device).repeat_interleave(view_count)


def style_weights(distances: Tensor) -> Tensor:
    flattened = distances.flatten()
    difference = (flattened.unsqueeze(0) - flattened.unsqueeze(1)).abs()
    return 1 / (1 + difference)


def acoustic_statistics(
    models: Stage2Models,
    batch: BeetleBatch,
) -> AcousticStatistics:
    with torch.no_grad():
        f0 = models.f0_extractor(batch.mel, batch.frame_mask)
    n = normalized_log_mel_energy(batch.mel, batch.frame_mask)
    f0_mask = batch.frame_mask[:, 0] & (f0 > 0)
    n_mask = batch.frame_mask[:, 0]
    f0_mean, f0_std = _masked_statistics(f0, f0_mask)
    n_mean, n_std = _masked_statistics(n, n_mask)
    return AcousticStatistics(f0_mean, f0_std, n_mean, n_std)


def _masked_statistics(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    numeric = mask.to(dtype=values.dtype)
    count = numeric.sum(dim=1).clamp_min(1)
    mean = (values * numeric).sum(dim=1) / count
    variance = ((values - mean.unsqueeze(1)).square() * numeric).sum(dim=1) / count
    return mean, torch.sqrt(variance.clamp_min(1e-5))
