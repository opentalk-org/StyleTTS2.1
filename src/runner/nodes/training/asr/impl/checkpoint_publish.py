from __future__ import annotations

from pathlib import Path

import yaml


def write_asr_bundle_artifacts(*, bundle_dir: Path, effective_config: dict) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = bundle_dir / "asr_train_config.yaml"
    cfg_path.write_text(yaml.safe_dump(effective_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    final_pth = bundle_dir / "final.pth"
    if not final_pth.is_file():
        raise ValueError("asr_publish_final_missing")
