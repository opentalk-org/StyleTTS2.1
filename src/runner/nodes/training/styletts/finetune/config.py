from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


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
    config["model_params"] = deepcopy(architecture["model_params"])


def build_config(
    *,
    log_dir: Path,
    train_list: str,
    validation_list: str,
    root_path: str,
    stream_from_buckets: bool,
    stream_plan_path: str,
    cache_dir: str,
    bucket_cache_budget_bytes: int,
    ood_texts: str,
    pretrained_model: Path,
    asr_config: dict[str, Any],
    asr_path: Path | None,
    f0_path: Path | None,
    plbert_config: dict[str, Any],
    plbert_path: Path | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_len: int,
    diff_epoch: int,
    joint_epoch: int,
    save_every_n_epochs: int,
    load_optimizer: bool,
    generator_checkpointing: bool,
    discriminators_checkpointing: bool,
    precision: str,
    slmadv_min_len: int,
    slmadv_max_len: int,
    slmadv_batch_samples: int,
    slmadv_scale: float,
    architecture_path: Path | None,
    multispeaker: bool | None,
    decoder_type: str | None,
    studio_publish: dict[str, Any],
    symbols: list[str],
    symbol_count: int,
) -> dict[str, Any]:
    config = deepcopy(load_yaml(BASE_YAML))
    config["log_dir"] = str(log_dir.resolve())
    config["epochs"] = int(epochs)
    config["batch_size"] = int(batch_size)
    config["max_len"] = int(max_len)
    config["save_freq"] = max(1, int(save_every_n_epochs))
    config["load_only_params"] = not load_optimizer
    config["data_params"] = {
        "train_data": str(Path(train_list).resolve()),
        "val_data": str(Path(validation_list).resolve()),
        "root_path": root_path,
        "OOD_data": str(Path(ood_texts).resolve()),
        "min_length": 50,
        "stream_from_buckets": bool(stream_from_buckets),
        "stream_plan_path": stream_plan_path,
        "cache_dir": cache_dir,
        "bucket_cache_budget_bytes": int(bucket_cache_budget_bytes),
    }
    config["pretrained_model"] = str(pretrained_model.resolve())
    config["ASR_config"] = asr_config
    config["ASR_path"] = _path_str(asr_path)
    config["F0_path"] = _path_str(f0_path)
    config["PLBERT_config"] = plbert_config
    config["PLBERT_path"] = _path_str(plbert_path)
    _apply_optimizer(config, learning_rate)
    _apply_stages(config, diff_epoch, joint_epoch)
    _apply_slm(config, slmadv_min_len, slmadv_max_len, slmadv_batch_samples, slmadv_scale)
    if architecture_path is not None:
        merge_architecture(architecture_path, config)
    _apply_model_overrides(config, multispeaker, decoder_type, generator_checkpointing, discriminators_checkpointing, symbol_count)
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


def _apply_stages(config: dict[str, Any], diff_epoch: int, joint_epoch: int) -> None:
    losses = config["loss_params"]
    losses["diff_epoch"] = int(diff_epoch)
    losses["joint_epoch"] = int(joint_epoch)


def _apply_slm(config: dict[str, Any], min_len: int, max_len: int, batch_samples: int, scale: float) -> None:
    slm = config["slmadv_params"]
    slm["min_len"] = int(min_len)
    slm["max_len"] = int(max_len)
    slm["batch_max_samples"] = int(batch_samples)
    slm["scale"] = float(scale)


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
