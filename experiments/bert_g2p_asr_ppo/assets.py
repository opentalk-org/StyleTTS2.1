from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from torch import Tensor

from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session

from .config import AssetConfig


@dataclass(frozen=True)
class BertAsset:
    weights: Path
    symbols: tuple[str, ...]
    languages: tuple[str, ...]
    encoder_state: dict[str, Tensor]
    language_weights: Tensor
    modality_weights: Tensor
    phoneme_head_weight: Tensor
    phoneme_head_bias: Tensor


@dataclass(frozen=True)
class ResolvedAssets:
    bert: BertAsset
    aligner_root: Path
    aligner_metadata: dict


def resolve_assets(config: AssetConfig) -> ResolvedAssets:
    with database_session() as session:
        bert_row = asset_crud.get_checkpoint(session, config.plbert_id)
        bert_root = asset_crud.get_checkpoint_path(session, config.plbert_id)
        aligner_row = asset_crud.get_checkpoint(session, config.aligner_checkpoint_id)
        aligner_root = asset_crud.get_checkpoint_path(session, config.aligner_checkpoint_id)
    if bert_row.type_ != "plbert":
        raise ValueError(f"expected plbert bucket asset, got {bert_row.type_}")
    if aligner_row.type_ != "styletts2":
        raise ValueError(f"expected styletts2 aligner checkpoint, got {aligner_row.type_}")
    weights = bert_root / str(bert_row.metadata_["model_file"])
    symbols = _json_tuple(bert_root / "tokenizer" / "phonemes.json", "symbols")
    languages = _json_tuple(bert_root / "tokenizer" / "languages.json", "languages")
    encoder_state: dict[str, Tensor] = {}
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            prefix = "encoder._orig_mod."
            if key.startswith(prefix):
                encoder_state[key.removeprefix(prefix)] = checkpoint.get_tensor(key)
        language_weights = checkpoint.get_tensor("language_embeddings.weight")
        modality_weights = checkpoint.get_tensor("modality_embeddings.weight")
        phoneme_head_weight = checkpoint.get_tensor("phoneme_head.weight")
        phoneme_head_bias = checkpoint.get_tensor("phoneme_head.bias")
    return ResolvedAssets(
        BertAsset(
            weights,
            symbols,
            languages,
            encoder_state,
            language_weights,
            modality_weights,
            phoneme_head_weight,
            phoneme_head_bias,
        ),
        aligner_root,
        {"type": aligner_row.type_, "metadata": aligner_row.metadata_},
    )


def _json_tuple(path: Path, key: str) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload[key]
    if not isinstance(values, list):
        raise TypeError(f"{path}:{key} must be a list")
    return tuple(str(value) for value in values)
