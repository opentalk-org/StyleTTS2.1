from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class LossConfig(StrictConfigModel):
    lambda_mel: float
    lambda_gen: float
    lambda_slm: float
    lambda_mono: float
    lambda_s2s: float
    lambda_F0: float
    lambda_norm: float
    lambda_dur: float
    lambda_ce: float
    lambda_sty: float
    lambda_diff: float
    diffusion_start_step: int = Field(ge=0)
    joint_start_step: int = Field(ge=0)


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
    validation_every_steps: int = Field(gt=0)
    checkpoint_every_steps: int = Field(gt=0)
    log_every_steps: int = Field(gt=0)
    profiling_enabled: bool
    distributed_processes: int = Field(gt=0)
    device: str
    batch_size: int = Field(gt=0)
    max_len: int = Field(gt=0)
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
    loss_params: LossConfig
    optimizer_params: OptimizerConfig
    slmadv_params: SlmAdversarialConfig
    studio_publish: dict[str, Any]
    symbols: list[str]

    @model_validator(mode="after")
    def validate_schedule(self) -> "TrainingConfig":
        losses = self.loss_params
        if losses.diffusion_start_step > losses.joint_start_step:
            raise ValueError(
                "diffusion_start_step must not exceed joint_start_step"
            )
        if losses.joint_start_step > self.total_steps:
            raise ValueError("joint_start_step must not exceed total_steps")
        return self


def load_training_config(path: str | Path) -> TrainingConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training config must be a mapping: {path}")
    return TrainingConfig.model_validate(payload)
