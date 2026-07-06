from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from runner.nodes.styletts_finetune.training.modules.asr.models import ASRCNN
from runner.nodes.styletts_finetune.training.state_dict_resize import merge_state_dict_with_dim0_resize

_ASR_N_TOKEN_DIM0_KEYS = frozenset({
    "ctc_linear.2.linear_layer.weight",
    "ctc_linear.2.linear_layer.bias",
    "asr_s2s.embedding.weight",
    "asr_s2s.project_to_n_symbols.weight",
    "asr_s2s.project_to_n_symbols.bias",
})


def _model_params_from_asr_yaml(config_path: str) -> dict[str, Any]:
    raw = Path(config_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        raise ValueError("asr_config_invalid")
    mp = cfg["model_params"]
    if not isinstance(mp, dict):
        raise ValueError("asr_model_params_missing")
    return mp


def init_ASR_model_from_config(config_path: str, *, target_n_token: int) -> ASRCNN:
    mp = _model_params_from_asr_yaml(config_path)
    return ASRCNN(
        input_dim=int(mp["input_dim"]),
        hidden_dim=int(mp["hidden_dim"]),
        n_token=int(target_n_token),
        n_layers=int(mp["n_layers"]),
        token_embedding_dim=int(mp["token_embedding_dim"]),
    )


def load_ASR_models(weights_path: str, config_path: str, *, target_n_token: int) -> ASRCNN:
    model = init_ASR_model_from_config(config_path, target_n_token=target_n_token)
    blob: dict[str, Any] | Any = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError("asr_weights_invalid")
    if "model" in blob:
        state = blob["model"]
    elif "net" in blob:
        state = blob["net"]
    else:
        raise KeyError("asr_weights_missing_model_or_net")
    if isinstance(state, dict):
        merged = merge_state_dict_with_dim0_resize(
            model,
            state,
            _ASR_N_TOKEN_DIM0_KEYS,
            error_scope="ASR",
        )
        model.load_state_dict(merged, strict=True)
    model.train()
    return model
