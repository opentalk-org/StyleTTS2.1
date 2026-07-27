from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    sample_rate: int = 22_050
    segment_samples: int = 8_192
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    mel_channels: int = 80
    f_min: float = 80.0
    f_max: float = 7_600.0


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    workers: int = 8
    learning_rate: float = 2e-4
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    learning_rate_decay: float = 0.999
    epochs: int = 3_100
    mel_weight: float = 45.0
    seed: int = 1_234


@dataclass(frozen=True)
class RunConfig:
    epochs: int
    checkpoint_interval: int
    validation_interval: int
    max_steps: int | None


SIGNAL = SignalConfig()
TRAINING = TrainingConfig()
