from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_REFERENCE_PATH = Path(__file__).resolve().parent / "reference_config.yml"


def load_asr_reference_yaml() -> dict[str, Any]:
    raw = _REFERENCE_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("asr_reference_config_invalid")
    return data


def build_effective_training_config(ref: dict[str, Any], *, symbols: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(ref)
    del cfg["ui_defaults"]
    mp = dict(cfg["model_params"])
    mp["n_token"] = len(symbols)
    cfg["model_params"] = mp
    dp = dict(cfg["data_params"])
    dp["phoneme_symbols"] = list(symbols)
    cfg["data_params"] = dp
    return cfg


def blank_index_from_config(cfg: dict[str, Any], *, symbol_to_idx: dict[str, int]) -> int:
    ch = str(cfg["data_params"]["ctc_blank_character"])[0]
    try:
        return int(symbol_to_idx[ch])
    except KeyError as exc:
        raise ValueError("asr_ctc_blank_not_in_vocab") from exc
