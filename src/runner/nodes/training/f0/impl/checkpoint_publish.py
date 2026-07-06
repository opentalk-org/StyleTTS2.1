from __future__ import annotations

from pathlib import Path


def validate_f0_checkpoint_folder(folder_path: Path) -> None:
    final_pth = folder_path / "final.pth"
    if not final_pth.is_file():
        raise ValueError("f0_publish_final_missing")
