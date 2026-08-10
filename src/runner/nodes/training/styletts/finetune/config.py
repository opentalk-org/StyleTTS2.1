from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .stages import TrainingStageSpec


DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_YAML = DATA_DIR / "base.yaml"
ASR_YAML = DATA_DIR / "asr.yml"
PLBERT_YAML = DATA_DIR / "plbert.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"yaml file must contain a mapping: {path}")
    return data


def merge_architecture(architecture_path: Path, config: dict[str, Any]) -> None:
    architecture = load_yaml(architecture_path)
    model_params = deepcopy(architecture["model_params"])
    model_params["alpha_flow"] = deepcopy(config["model_params"]["alpha_flow"])
    if "prosody_quantizer" not in model_params:
        model_params["prosody_quantizer"] = deepcopy(
            config["model_params"]["prosody_quantizer"]
        )
    config["model_params"] = model_params


def build_config(
    *,
    log_dir: Path,
    dataset_id: str,
    validation_samples: int,
    pretrained_model: Path | None,
    asr_config: dict[str, Any],
    asr_path: Path | None,
    f0_path: Path | None,
    plbert_config: dict[str, Any],
    plbert_path: Path | None,
    total_steps: int,
    seed: int,
    learning_rate: float,
    training_stages: list[TrainingStageSpec],
    validation_every_steps: int,
    checkpoint_every_steps: int,
    log_every_steps: int,
    profiling_enabled: bool,
    distributed_processes: int,
    load_optimizer: bool,
    generator_checkpointing: bool,
    discriminators_checkpointing: bool,
    precision: str,
    architecture_path: Path | None,
    multispeaker: bool | None,
    decoder_type: str | None,
    studio_publish: dict[str, Any],
    symbols: list[str],
    symbol_count: int,
) -> dict[str, Any]:
    config = deepcopy(load_yaml(BASE_YAML))
    config["log_dir"] = str(log_dir.resolve())
    config["total_steps"] = int(total_steps)
    config["seed"] = int(seed)
    config["validation_every_steps"] = int(validation_every_steps)
    config["checkpoint_every_steps"] = int(checkpoint_every_steps)
    config["log_every_steps"] = int(log_every_steps)
    config["profiling_enabled"] = profiling_enabled
    config["distributed_processes"] = int(distributed_processes)
    config["load_only_params"] = not load_optimizer
    config["data_params"] = {
        "dataset_id": dataset_id,
        "validation_samples": validation_samples,
    }
    config["pretrained_model"] = _path_str(pretrained_model)
    config["ASR_config"] = asr_config
    config["ASR_path"] = _path_str(asr_path)
    config["F0_path"] = _path_str(f0_path)
    config["PLBERT_config"] = plbert_config
    config["PLBERT_path"] = _path_str(plbert_path)
    _apply_optimizer(config, learning_rate)
    config["training_stages"] = [
        stage.model_dump(mode="json")
        for stage in training_stages
    ]
    if architecture_path is not None:
        merge_architecture(architecture_path, config)
    _apply_model_overrides(config, multispeaker, decoder_type, generator_checkpointing, discriminators_checkpointing, symbol_count)
    if architecture_path is None or plbert_path is not None:
        languages = plbert_config.get("languages", [])
        config["model_params"]["language_count"] = max(2, len(languages) + 1)
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError("precision must be fp16, bf16, or fp32")
    config["precision"] = precision
    config["studio_publish"] = studio_publish
    config["symbols"] = symbols
    return config


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _path_str(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve())


def _apply_optimizer(config: dict[str, Any], learning_rate: float) -> None:
    optimizer = config["optimizer_params"]
    optimizer["lr"] = float(learning_rate)
    optimizer["bert_lr"] = max(1e-6, float(learning_rate) * 0.1)
    optimizer["ft_lr"] = float(learning_rate)


def _apply_model_overrides(
    config: dict[str, Any],
    multispeaker: bool | None,
    decoder_type: str | None,
    generator_checkpointing: bool,
    discriminators_checkpointing: bool,
    symbol_count: int,
) -> None:
    params = config["model_params"]
    if multispeaker is not None:
        params["multispeaker"] = bool(multispeaker)
    if decoder_type is not None:
        if decoder_type not in ("hifigan", "istftnet"):
            raise ValueError("decoder type must be hifigan or istftnet")
        params["decoder"]["type"] = decoder_type
    params["decoder"]["gradient_checkpointing"] = bool(generator_checkpointing)
    params["discriminators_checkpointing"] = bool(discriminators_checkpointing)
    params["n_token"] = int(symbol_count)
    params.setdefault(
        "alpha_flow",
        {
            "transition_start": 0,
            "transition_end": 10000,
            "temperature": 25.0,
            "flow_matching_ratio": 0.5,
            "conditional_dropout": 0.1,
        },
    )
