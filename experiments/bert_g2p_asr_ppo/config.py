from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class AssetConfig:
    plbert_id: UUID = UUID("f4109860-a92d-47a9-9717-1d2f2febac4b")
    aligner_checkpoint_id: UUID = UUID("4a3f31d4-6fda-463f-9f8a-f9d85cbf84a9")


@dataclass(frozen=True)
class DataConfig:
    host: str = "hetzner-storagebox"
    remote_train_dir: str = "/home/lang-pl-bert/data/pl_bert_multilingual"
    remote_validation_dir: str = "/home/lang-pl-bert/data/pl_bert_validation"
    cache_dir: Path = Path("data/experiments/bert_g2p_asr_ppo/parquet")
    train_files: int = 1
    validation_files: int = 64
    max_text_bytes: int = 256
    max_phonemes: int = 384
    language: str = "pl"
    packed_text_bytes: int = 192


@dataclass(frozen=True)
class SftConfig:
    steps: int = 10_000
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    log_interval: int = 10
    checkpoint_interval: int = 1000
    validation_interval: int = 250
    validation_batches: int = 16
    autoregressive_validation_batches: int = 1
    validation_generation_tokens: int = 128


@dataclass(frozen=True)
class PpoConfig:
    dataset_id: UUID = UUID("e25b39ac-3400-4f9f-9fac-3e9c94e1a92b")
    steps: int = 500
    batch_size: int = 4
    rollout_tokens: int = 384
    update_epochs: int = 2
    learning_rate: float = 1e-6
    clip_ratio: float = 0.2
    kl_coefficient: float = 0.2
    entropy_coefficient: float = 0.001
    log_interval: int = 5
    validation_interval: int = 25
    validation_batches: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("data/experiments/bert_g2p_asr_ppo/runs")
    run_name: str = "two_bert_g2p_asr_ppo"
    seed: int = 20260828
    assets: AssetConfig = field(default_factory=AssetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sft: SftConfig = field(default_factory=SftConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)

    def payload(self) -> dict:
        return asdict(self)
