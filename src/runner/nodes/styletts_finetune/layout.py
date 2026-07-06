from __future__ import annotations

from pathlib import Path


def architecture_yaml(library_root: Path) -> Path:
    root = library_root.resolve()
    if not root.is_dir():
        raise ValueError("styletts library root is not a directory")
    config_path = root / "config.yml"
    if config_path.is_file():
        return config_path
    raise ValueError("styletts architecture config.yml not found")


def latest_weight(library_root: Path) -> Path:
    root = library_root.resolve()
    weights = [path for path in root.rglob("*.pth") if path.is_file()]
    if not weights:
        raise ValueError("styletts checkpoint folder does not contain .pth weights")
    return max(weights, key=lambda path: path.stat().st_mtime)
