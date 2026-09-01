from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from experiments.bert_g2p_asr_ppo.config import AssetConfig, DataConfig


@dataclass(frozen=True)
class ModelConfig:
    encoder_hidden_size: int = 512
    encoder_layers: int = 2
    predictor_hidden_size: int = 512
    predictor_layers: int = 2
    joiner_hidden_size: int = 512
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10_000
    batch_size: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    log_interval: int = 10
    validation_interval: int = 250
    validation_batches: int = 16
    checkpoint_interval: int = 1000
    max_symbols_per_timestep: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("data/experiments/bert_bilstm_rnnt/runs")
    run_name: str = "bert_bilstm_rnnt_g2p"
    seed: int = 20260831
    assets: AssetConfig = field(default_factory=AssetConfig)
    data: DataConfig = field(default_factory=partial(DataConfig, language="all", train_files=16))
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def payload(self) -> dict:
        return asdict(self)
