from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    sample_rate: int = 24_000
    segment_samples: int = 24_576
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    mel_channels: int = 80
    f_min: float = 0.0
    f_max: float = 12_000.0
    jdc_f_max: float = 8_000.0


@dataclass(frozen=True)
class GeneratorConfig:
    sampling_rate: int = 24_000
    resblock: str = "1"
    final_stage_resblock_bottleneck: int = 2
    upsample_rates: tuple[int, ...] = (8, 8)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16)
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    gen_istft_n_fft: int = 16
    gen_istft_hop_size: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    workers: int = 8
    weight_norm_interval: int = 100
    learning_rate: float = 2e-4
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    generator_gradient_clip_norm: float = 5_000.0
    discriminator_gradient_clip_norm: float = 100.0
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
GENERATOR = GeneratorConfig()
TRAINING = TrainingConfig()
