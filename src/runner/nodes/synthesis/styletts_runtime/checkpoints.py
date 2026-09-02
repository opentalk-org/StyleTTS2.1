from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from runner.nodes.models import CheckpointRef
from shared.db import database_session
from shared.db.assets import crud as asset_crud


F0_SUFFIXES = frozenset({".t7", ".pth", ".pt", ".ckpt"})
ASR_CONFIG_SUFFIXES = frozenset({".yml", ".yaml"})
ASR_WEIGHT_SUFFIXES = frozenset({".pth", ".pt", ".ckpt"})
DATA_DIR = Path(__file__).resolve().parents[2] / "training" / "styletts" / "finetune" / "data"


@dataclass(frozen=True)
class ResolvedCheckpoint:
    id: UUID
    name: str
    root: Path
    type_: str
    metadata: dict[str, Any]


def resolve_main_checkpoint(value: CheckpointRef) -> ResolvedCheckpoint:
    metadata = _checkpoint_metadata(value)
    type_ = str(value.metadata["type"])
    if type_ != "styletts2":
        raise ValueError("finetune_test_checkpoint_invalid")
    return ResolvedCheckpoint(value.checkpoint_id, value.name, value.path, type_, metadata)


def resolve_slot_checkpoint(checkpoint_id: UUID, expected_type: str) -> ResolvedCheckpoint:
    with database_session() as session:
        item = asset_crud.get_checkpoint(checkpoint_id)
        root = asset_crud.get_checkpoint_path(session, checkpoint_id)
    if item.type_ != expected_type:
        raise ValueError(f"checkpoint {checkpoint_id} has type {item.type_}, expected {expected_type}")
    return ResolvedCheckpoint(item.id, item.name, root, item.type_, item.metadata_)


def latest_weight(root: Path) -> Path:
    weights = [path for path in root.resolve().rglob("*.pth") if path.is_file()]
    if not weights:
        raise ValueError("styletts checkpoint folder does not contain .pth weights")
    return max(weights, key=lambda path: path.stat().st_mtime)


def resolve_symbols(metadata: dict[str, Any]) -> list[str]:
    from runner.nodes.text.runtime.symbols import DEFAULT_STYLETTS_SYMBOLS

    raw = metadata.get("symbols")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    if isinstance(raw, str) and raw:
        return list(raw)
    return [str(symbol) for symbol in DEFAULT_STYLETTS_SYMBOLS]


def resolve_asr_payload(checkpoint_id: UUID | None, target_symbols: list[str]) -> tuple[dict[str, Any], str | None]:
    if checkpoint_id is None:
        config = _load_yaml(DATA_DIR / "asr.yml")
        config["model_params"]["n_token"] = len(target_symbols)
        return config, None
    bundle = resolve_slot_checkpoint(checkpoint_id, "asr_bundle")
    config_path, weights_path = asr_bundle_config_and_weights(bundle.root)
    config = _load_yaml(config_path)
    if bundle.metadata["symbols"] != target_symbols:
        config["model_params"]["n_token"] = len(target_symbols)
    return config, str(weights_path.resolve())


def resolve_f0_path(checkpoint_id: UUID | None, inner_filename: str) -> str | None:
    if checkpoint_id is None:
        return None
    bundle = resolve_slot_checkpoint(checkpoint_id, "f0_model")
    inner = inner_filename.strip() or None
    return str(f0_weight_path_in_slot_dir(bundle.root, inner).resolve())


def resolve_plbert_payload(checkpoint_id: UUID | None, target_symbols: list[str]) -> tuple[dict[str, Any], str | None]:
    if checkpoint_id is None:
        config = _load_yaml(DATA_DIR / "plbert.yml")
        config["model_params"]["vocab_size"] = len(target_symbols)
        return config, None
    bundle = resolve_slot_checkpoint(checkpoint_id, "plbert")
    if not plbert_bundle_dir_valid(bundle.root):
        raise ValueError("plbert_dir_invalid")
    config = _load_yaml(bundle.root / "config.yml")
    if bundle.metadata["symbols"] != target_symbols:
        config["model_params"]["vocab_size"] = len(target_symbols)
    weights = next((path for path in bundle.root.glob("*.t7") if path.is_file()), None)
    if weights is None:
        raise ValueError("plbert_path_not_found")
    return config, str(weights.resolve())


def asr_bundle_config_and_weights(bundle_dir: Path) -> tuple[Path, Path]:
    if not bundle_dir.is_dir():
        raise ValueError("finetune_asr_bundle_invalid")
    files = [path for path in bundle_dir.iterdir() if path.is_file()]
    configs = sorted(path for path in files if path.suffix.lower() in ASR_CONFIG_SUFFIXES)
    weights = sorted(path for path in files if path.suffix.lower() in ASR_WEIGHT_SUFFIXES)
    if len(configs) != 1:
        raise ValueError("finetune_asr_config_ambiguous")
    if len(weights) != 1:
        raise ValueError("finetune_asr_weights_ambiguous")
    return configs[0], weights[0]


def f0_weight_path_in_slot_dir(base: Path, inner_filename: str | None) -> Path:
    if not base.is_dir():
        raise ValueError("finetune_f0_invalid")
    candidates = sorted(path for path in base.iterdir() if path.is_file() and path.suffix.lower() in F0_SUFFIXES)
    if not candidates:
        raise ValueError("finetune_f0_missing")
    if len(candidates) == 1:
        return _single_f0_candidate(candidates[0], inner_filename)
    return _chosen_f0_candidate(base, inner_filename)


def plbert_bundle_dir_valid(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {item.name for item in path.iterdir() if item.is_file()}
    return "config.yml" in names and any(name.endswith(".t7") for name in names)


def _checkpoint_metadata(value: CheckpointRef) -> dict[str, Any]:
    return value.metadata["metadata"]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"yaml file must contain a mapping: {path}")
    return data


def _single_f0_candidate(candidate: Path, inner_filename: str | None) -> Path:
    if inner_filename and inner_filename.strip() and inner_filename.strip() != candidate.name:
        raise ValueError("finetune_f0_inner_mismatch")
    return candidate


def _chosen_f0_candidate(base: Path, inner_filename: str | None) -> Path:
    if not inner_filename or not inner_filename.strip():
        raise ValueError("finetune_f0_choice_required")
    name = Path(inner_filename.strip()).name
    if name != inner_filename.strip():
        raise ValueError("finetune_f0_inner_invalid")
    chosen = (base / name).resolve()
    if not chosen.is_file() or chosen.parent != base.resolve():
        raise ValueError("finetune_f0_inner_invalid")
    if chosen.suffix.lower() not in F0_SUFFIXES:
        raise ValueError("finetune_f0_inner_invalid")
    return chosen
