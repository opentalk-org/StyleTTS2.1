import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .training import BeetleConfig


def load_config(path: str | Path) -> BeetleConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping: {config_path}")
    return BeetleConfig.model_validate(raw)


def config_fingerprint(config: BeetleConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
