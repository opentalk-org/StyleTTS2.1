from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from runner.nodes.text.runtime.symbols import PAD_SYMBOL

_REFERENCE_PATH = Path(__file__).resolve().parent / "reference_config.yml"


def load_asr_reference_yaml() -> dict[str, Any]:
    raw = _REFERENCE_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("asr_reference_config_invalid")
    return data


def build_effective_training_config(ref: dict[str, Any], *, symbols: list[str]) -> dict[str, Any]:
    if not symbols or symbols[0] != PAD_SYMBOL:
        raise ValueError(f"phoneme alphabet must start with {PAD_SYMBOL!r}")
    cfg = copy.deepcopy(ref)
    mp = dict(cfg["model_params"])
    mp["n_token"] = len(symbols)
    cfg["model_params"] = mp
    dp = dict(cfg["data_params"])
    dp["phoneme_symbols"] = list(symbols)
    cfg["data_params"] = dp
    return cfg
