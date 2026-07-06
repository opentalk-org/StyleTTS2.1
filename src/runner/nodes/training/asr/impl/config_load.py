from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from runner.nodes.text.runtime.symbols import default_styletts_testing_phoneme_symbols

_REFERENCE_PATH = Path(__file__).resolve().parent / "reference_config.yml"


def reference_config_path() -> Path:
    return _REFERENCE_PATH


def load_asr_reference_yaml() -> dict[str, Any]:
    raw = _REFERENCE_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("asr_reference_config_invalid")
    return data


def resolve_phoneme_symbols(ref: dict[str, Any]) -> list[str]:
    dp = ref.get("data_params")
    if not isinstance(dp, dict):
        dp = {}
    raw = dp.get("phoneme_symbols")
    if raw is None or (isinstance(raw, list) and len(raw) == 0):
        return default_styletts_testing_phoneme_symbols()
    if not isinstance(raw, list):
        raise ValueError("asr_phoneme_symbols_invalid")
    return [str(x) for x in raw]


def ui_defaults_from_reference(ref: dict[str, Any]) -> dict[str, Any]:
    ud = ref.get("ui_defaults")
    if not isinstance(ud, dict):
        return {}
    return dict(ud)


def build_effective_training_config(ref: dict[str, Any], *, symbols: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(ref)
    cfg.pop("ui_defaults", None)
    mp = dict(cfg.get("model_params") or {})
    mp["n_token"] = len(symbols)
    cfg["model_params"] = mp
    dp = dict(cfg.get("data_params") or {})
    dp["phoneme_symbols"] = list(symbols)
    cfg["data_params"] = dp
    return cfg


def blank_index_from_config(cfg: dict[str, Any], *, symbol_to_idx: dict[str, int]) -> int:
    dp = cfg.get("data_params") or {}
    ch = " "
    if isinstance(dp, dict):
        raw = dp.get("ctc_blank_character")
        if isinstance(raw, str) and raw:
            ch = raw[:1]
    try:
        return int(symbol_to_idx[ch])
    except KeyError as exc:
        raise ValueError("asr_ctc_blank_not_in_vocab") from exc
