from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .architecture import ArchitectureConfig, AudioConfig, StrictConfigModel
from .data import DataConfig


class Precision(StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class OptimizerConfig(StrictConfigModel):
    learning_rate: float = Field(gt=0)
    beta1: float = Field(ge=0, lt=1)
    beta2: float = Field(ge=0, lt=1)
    epsilon: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    decay_steps: int = Field(gt=0)
    minimum_learning_rate_ratio: float = Field(ge=0, le=1)
    maximum_gradient_norm: float = Field(gt=0)


class ScheduledWeight(StrictConfigModel):
    value: float = Field(ge=0)
    start_step: int = Field(ge=0)
    warmup_steps: int = Field(ge=0)


class LossWeights(StrictConfigModel):
    encoder_kl: ScheduledWeight
    f0: ScheduledWeight
    n: ScheduledWeight
    reconstruction: ScheduledWeight
    discriminator: ScheduledWeight
    generator_adversarial: ScheduledWeight
    feature_matching: ScheduledWeight
    duration_flow: ScheduledWeight
    latent_flow: ScheduledWeight
    shortcut: ScheduledWeight
    align_s2s: ScheduledWeight
    align_mono: ScheduledWeight
    align_ctc: ScheduledWeight
    voice_contrastive: ScheduledWeight
    voice_ge2e: ScheduledWeight
    style_contrastive: ScheduledWeight
    style_ge2e: ScheduledWeight
    style_speaker_adversarial: ScheduledWeight
    style_statistics: ScheduledWeight
    style_reencoding: ScheduledWeight


class StageConfig(StrictConfigModel):
    batch_size: int = Field(gt=0)
    accumulation_steps: int = Field(gt=0)
    precision: Precision
    generator_optimizer: OptimizerConfig
    discriminator_optimizer: OptimizerConfig | None
    losses: LossWeights


class RuntimeConfig(StrictConfigModel):
    seed: int = Field(ge=0)
    device: str = Field(min_length=1)
    log_every_steps: int = Field(gt=0)


class ComplexityConfig(StrictConfigModel):
    minimum_inference_parameters: int = Field(gt=0)
    maximum_inference_parameters: int = Field(gt=0)
    latent_audio_max_gflops_per_second: float = Field(gt=0)
    benchmark_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_parameter_bounds(self) -> "ComplexityConfig":
        if self.minimum_inference_parameters >= self.maximum_inference_parameters:
            raise ValueError(
                "minimum_inference_parameters must be below "
                "maximum_inference_parameters"
            )
        return self


class ValidationConfig(StrictConfigModel):
    every_steps: int = Field(gt=0)
    sample_count: int = Field(gt=0)


class CheckpointConfig(StrictConfigModel):
    every_steps: int = Field(gt=0)
    keep_last: int = Field(gt=0)


class Stage2ObjectiveConfig(StrictConfigModel):
    contrastive_temperature: float = Field(gt=0)
    reversal_scale: float = Field(ge=0)
    consistency_cosine_weight: float = Field(ge=0)
    consistency_mse_weight: float = Field(ge=0)


class BeetleConfig(StrictConfigModel):
    audio: AudioConfig
    architecture: ArchitectureConfig
    complexity: ComplexityConfig
    data: DataConfig
    runtime: RuntimeConfig
    validation: ValidationConfig
    checkpoint: CheckpointConfig
    stage2_objective: Stage2ObjectiveConfig
    stage1: StageConfig
    stage2: StageConfig
    stage3: StageConfig

    @model_validator(mode="before")
    @classmethod
    def reject_dataset_pass_vocabulary(cls, value: Any) -> Any:
        def visit(item: Any, path: tuple[str, ...]) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    if "epoch" in str(key).lower():
                        location = ".".join((*path, str(key)))
                        raise ValueError(f"epoch fields are forbidden: {location}")
                    visit(child, (*path, str(key)))
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, (*path, str(index)))

        visit(value, ())
        return value

    @model_validator(mode="after")
    def validate_composition(self) -> "BeetleConfig":
        if self.audio.hop_length != 300:
            raise ValueError("hop_length must be exactly 300")
        if self.architecture.posterior.mel_channels != self.audio.mel_channels:
            raise ValueError("posterior mel_channels must match audio mel_channels")
        if self.architecture.generator.output_hop() != self.audio.hop_length:
            raise ValueError("generator output geometry must match hop_length")
        if self.stage1.discriminator_optimizer is None:
            raise ValueError("stage1 requires discriminator_optimizer")
        if self.stage2.discriminator_optimizer is not None:
            raise ValueError("stage2 must not configure discriminator_optimizer")
        if self.stage3.discriminator_optimizer is None:
            raise ValueError("stage3 requires discriminator_optimizer")
        return self
