from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .architecture import ArchitectureConfig, AudioConfig, StrictConfigModel
from .data import DataConfig
from .validation import ValidationConfig


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


class TrainingConfig(StrictConfigModel):
    batch_size: int = Field(gt=0)
    accumulation_steps: int = Field(gt=0)
    total_steps: int = Field(gt=0)
    validation_every_steps: int = Field(gt=0)
    overfit_validation_recording: bool
    complex_reconstruction_steps: int = Field(ge=0)
    full_audio_ratio: float = Field(ge=0, le=1)
    precision: Precision
    acoustic_prediction: ScheduledWeight
    generator_optimizer: OptimizerConfig
    discriminator_optimizer: OptimizerConfig
    losses: LossWeights


class RuntimeConfig(StrictConfigModel):
    seed: int = Field(ge=0)
    device: str = Field(min_length=1)
    compile: bool
    compile_frame_count: int | None
    log_every_steps: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_compilation(self) -> "RuntimeConfig":
        if self.compile != (self.compile_frame_count is not None):
            raise ValueError(
                "compile_frame_count must be set exactly when compilation is enabled"
            )
        if self.compile_frame_count is not None and self.compile_frame_count % 2:
            raise ValueError("compile_frame_count must be even")
        return self


class AdversarialConfig(StrictConfigModel):
    segment_samples: int = Field(gt=0)


class CheckpointConfig(StrictConfigModel):
    every_steps: int = Field(gt=0)
    keep_last: int = Field(gt=0)


class ConditioningObjectiveConfig(StrictConfigModel):
    contrastive_temperature: float = Field(gt=0)
    reversal_scale: float = Field(ge=0)
    consistency_cosine_weight: float = Field(ge=0)
    consistency_mse_weight: float = Field(ge=0)


class BeetleConfig(StrictConfigModel):
    audio: AudioConfig
    architecture: ArchitectureConfig
    data: DataConfig
    validation: ValidationConfig
    runtime: RuntimeConfig
    adversarial: AdversarialConfig
    checkpoint: CheckpointConfig
    conditioning_objective: ConditioningObjectiveConfig
    training: TrainingConfig

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
        if self.audio.hop_length != 256:
            raise ValueError("hop_length must be exactly 256")
        if self.architecture.posterior.mel_channels != self.audio.mel_channels:
            raise ValueError("posterior mel_channels must match audio mel_channels")
        if self.architecture.generator.output_hop() != self.audio.hop_length:
            raise ValueError("generator output geometry must match hop_length")
        posterior = self.architecture.posterior
        receptive_field = posterior.receptive_field_mel_frames()
        if receptive_field % 2:
            raise ValueError("posterior receptive field must be even")
        target_frames = self.adversarial.segment_samples // self.audio.hop_length
        encoder_frames = target_frames + receptive_field
        if self.runtime.compile and self.runtime.compile_frame_count != encoder_frames:
            raise ValueError(
                "compile_frame_count must match contextual encoder frames"
            )
        segment_samples = self.adversarial.segment_samples
        if segment_samples % self.audio.hop_length:
            raise ValueError("adversarial segment_samples must divide by hop_length")
        segment_frames = segment_samples // self.audio.hop_length
        if segment_frames % self.architecture.posterior.downsample_rate:
            raise ValueError(
                "adversarial segment_samples must align with posterior downsampling"
            )
        if self.training.overfit_validation_recording:
            training_ids = self.data.selection.audio_file_ids
            validation_ids = self.validation.audio_file_ids
            if self.training.batch_size != 1:
                raise ValueError("validation-recording overfit requires batch_size 1")
            if self.validation.sample_count != 1:
                raise ValueError("validation-recording overfit requires one sample")
            if len(training_ids) != 1 or training_ids != validation_ids:
                raise ValueError(
                    "validation-recording overfit requires one matching training "
                    "and validation audio ID"
                )
        if self.training.complex_reconstruction_steps > self.training.total_steps:
            raise ValueError(
                "complex reconstruction steps cannot exceed total training steps"
            )
        return self
