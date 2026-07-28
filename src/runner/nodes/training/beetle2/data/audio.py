import io
import random
from dataclasses import dataclass

import soundfile
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

from ..config.data import AugmentationConfig
from ..losses.acoustic import LogMelSpectrogram
from .records import SegmentKey
from shared.db.audio.ranges.wav import WavClip


@dataclass(frozen=True)
class ProcessedAudio:
    waveform: Tensor
    mel: Tensor
    jdc_mel: Tensor


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        mel_channels: int,
        f_min: float,
        f_max: float,
        jdc_f_max: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_transform = LogMelSpectrogram(
            sample_rate,
            n_fft,
            hop_length,
            win_length,
            mel_channels,
            f_min,
            f_max,
        )
        self.jdc_mel_transform = LogMelSpectrogram(
            sample_rate,
            n_fft,
            hop_length,
            win_length,
            mel_channels,
            f_min,
            jdc_f_max,
        )

    def decode(self, clip: WavClip, key: SegmentKey) -> ProcessedAudio:
        waveform = self.decode_waveform(clip, key)
        if waveform.shape[-1] < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.shape[-1]))
        mel = self.mel_transform(waveform).squeeze(0)
        jdc_mel = self.jdc_mel_transform(waveform).squeeze(0)
        target_samples = mel.shape[-1] * self.hop_length
        waveform = _fit_length(waveform, target_samples)
        if not torch.isfinite(mel).all():
            raise ValueError(f"non-finite mel features for {key}")
        if not torch.isfinite(jdc_mel).all():
            raise ValueError(f"non-finite JDC mel features for {key}")
        return ProcessedAudio(
            waveform.contiguous(),
            mel.contiguous(),
            jdc_mel.contiguous(),
        )

    def decode_waveform(self, clip: WavClip, key: SegmentKey) -> Tensor:
        try:
            values, sample_rate = soundfile.read(
                io.BytesIO(clip.data),
                dtype="float32",
                always_2d=True,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"unable to decode WAV for {key}: {error}") from error
        if sample_rate != clip.sample_rate:
            raise ValueError(f"WAV sample-rate metadata mismatch for {key}")
        waveform = torch.from_numpy(values).transpose(0, 1).mean(dim=0, keepdim=True)
        if not torch.isfinite(waveform).all():
            raise ValueError(f"non-finite WAV samples for {key}")
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.contiguous()

    def augment(
        self,
        waveform: Tensor,
        seed: int,
        config: AugmentationConfig,
    ) -> Tensor:
        rng = random.Random(seed)
        stretch = rng.uniform(config.time_stretch_min, config.time_stretch_max)
        semitones = rng.uniform(
            config.pitch_shift_min_semitones,
            config.pitch_shift_max_semitones,
        )
        gain_db = rng.uniform(config.gain_min_db, config.gain_max_db)
        original_length = waveform.shape[-1]
        stretched_length = max(1, round(original_length * stretch))
        output = F.interpolate(
            waveform.unsqueeze(0),
            size=stretched_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        if semitones != 0:
            output = torchaudio.functional.pitch_shift(output, self.sample_rate, semitones)
        output = output * (10 ** (gain_db / 20))
        return _fit_length(output, original_length).clamp(-1, 1).contiguous()


def _fit_length(waveform: Tensor, length: int) -> Tensor:
    if waveform.shape[-1] < length:
        return F.pad(waveform, (0, length - waveform.shape[-1]))
    return waveform[..., :length]
