from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...stages import TrainingStageSpec


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictConfigModel):
    train_data: str
    val_data: str
    root_path: str
    OOD_data: str
    min_length: int = Field(gt=0)
    stream_from_buckets: bool
    stream_plan_path: str
    cache_dir: str
    bucket_cache_budget_bytes: int = Field(ge=0)


class SpectrogramConfig(StrictConfigModel):
    n_fft: int = Field(gt=0)
    win_length: int = Field(gt=0)
    hop_length: int = Field(gt=0)


class PreprocessConfig(StrictConfigModel):
    sr: int = Field(gt=0)
    spect_params: SpectrogramConfig


class OptimizerConfig(StrictConfigModel):
    lr: float = Field(gt=0)
    bert_lr: float = Field(gt=0)
    ft_lr: float = Field(gt=0)


class SlmAdversarialConfig(StrictConfigModel):
    min_len: int = Field(gt=0)
    max_len: int = Field(gt=0)
    batch_max_samples: int = Field(ge=0)
    iter: int = Field(gt=0)
    thresh: float = Field(gt=0)
    scale: float = Field(gt=0)
    sig: float = Field(gt=0)


class TrainingConfig(StrictConfigModel):
    log_dir: str
    total_steps: int = Field(gt=0)
    seed: int = Field(ge=0)
    validation_every_steps: int = Field(gt=0)
    checkpoint_every_steps: int = Field(gt=0)
    log_every_steps: int = Field(gt=0)
    profiling_enabled: bool
    distributed_processes: int = Field(gt=0)
    device: str
    second_stage_load_pretrained: bool
    load_only_params: bool
    precision: Literal["fp16", "bf16", "fp32"]
    pretrained_model: str | None
    ASR_config: dict[str, Any]
    ASR_path: str | None
    F0_path: str | None
    PLBERT_config: dict[str, Any]
    PLBERT_path: str | None
    model_params: dict[str, Any]
    data_params: DataConfig
    preprocess_params: PreprocessConfig
    optimizer_params: OptimizerConfig
    slmadv_params: SlmAdversarialConfig
    studio_publish: dict[str, Any]
    symbols: list[str]
    training_stages: list[TrainingStageSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_schedule(self) -> "TrainingConfig":
        scheduled_steps = sum(stage.steps for stage in self.training_stages)
        if scheduled_steps != self.total_steps:
            raise ValueError(
                "training stage steps must sum to total_steps"
            )
        return self


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training config must be a mapping: {path}")
    return TrainingConfig.model_validate(payload)
