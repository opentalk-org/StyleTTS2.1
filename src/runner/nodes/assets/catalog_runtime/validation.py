from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


F0_SUFFIXES = frozenset({".t7", ".pth", ".pt", ".ckpt"})
ASR_CONFIG_SUFFIXES = frozenset({".yml", ".yaml"})
ASR_WEIGHT_SUFFIXES = frozenset({".pth", ".pt", ".ckpt"})
STYLE_WEIGHT_SUFFIXES = frozenset({".pth", ".pt", ".ckpt"})
MIN_OFFICIAL_CHECKPOINT_BYTES = 50_000_000
DEFAULT_SYMBOLS = (
    "a b c d e f g h i j k l m n o p q r s t u v w x y z "
    "ɑ æ ə ɛ ɪ ʊ ʌ ɔ θ ð ʃ ʒ ŋ tʃ dʒ aɪ aʊ eɪ oʊ ɔɪ ɝ ɚ "
    "ˈ ˌ ː . , ? ! ' \" ( ) -"
)


def default_symbols_metadata() -> dict[str, Any]:
    return {"symbols": [part for part in DEFAULT_SYMBOLS.split(" ") if part]}


def styletts_checkpoint_metadata(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("styletts_official_config_invalid")
    model_params = cfg["model_params"]
    decoder = model_params["decoder"]
    return {
        "multispeaker": model_params["multispeaker"],
        "decoder_type": str(decoder["type"]).strip().lower(),
        **default_symbols_metadata(),
    }


def asr_bundle_valid(path: Path) -> bool:
    try:
        files = _files(path)
    except ValueError:
        return False
    configs = [item for item in files if item.suffix.lower() in ASR_CONFIG_SUFFIXES]
    weights = [item for item in files if item.suffix.lower() in ASR_WEIGHT_SUFFIXES]
    return len(configs) == 1 and len(weights) == 1


def f0_bundle_valid(path: Path) -> bool:
    try:
        files = _files(path)
    except ValueError:
        return False
    return any(item.suffix.lower() in F0_SUFFIXES for item in files)


def plbert_bundle_valid(path: Path) -> bool:
    try:
        names = {item.name for item in _files(path)}
    except ValueError:
        return False
    return "config.yml" in names and any(name.endswith(".t7") for name in names)


def official_styletts_bundle_valid(path: Path) -> bool:
    try:
        files = _files(path)
    except ValueError:
        return False
    weights = [item for item in files if item.suffix.lower() in STYLE_WEIGHT_SUFFIXES]
    configs = [item for item in files if item.name == "config.yml"]
    return len(configs) == 1 and any(item.stat().st_size >= MIN_OFFICIAL_CHECKPOINT_BYTES for item in weights)


def _files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise ValueError("catalog_bundle_path_invalid")
    return [item for item in path.iterdir() if item.is_file()]
