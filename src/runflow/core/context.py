from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ExecutionContext:
    run_id: str = field(default_factory=lambda: f"run_{uuid4().hex[:8]}")
    work_dir: Path = Path("work")
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    device: str = "cuda"
    config: dict[str, Any] = field(default_factory=dict)
    # Generic source items for schedulers. They may be paths, URLs, IDs, records, etc.
    input_items: list[Any] = field(default_factory=list)
    window_index: int = 0
    current_window_items: list[Any] = field(default_factory=list)

    # Convenience/backward-compatible aliases for path-based examples.
    @property
    def input_paths(self) -> list[Path]:
        return [Path(item) for item in self.input_items]

    @input_paths.setter
    def input_paths(self, value: list[Path]) -> None:
        self.input_items = list(value)

    @property
    def current_window_paths(self) -> list[Path]:
        return [Path(item) for item in self.current_window_items]

    @current_window_paths.setter
    def current_window_paths(self, value: list[Path]) -> None:
        self.current_window_items = list(value)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def window_dir(self) -> Path:
        path = self.work_dir / self.run_id / f"window_{self.window_index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def node_dir(self, node_id: str) -> Path:
        path = self.window_dir / node_id
        path.mkdir(parents=True, exist_ok=True)
        return path
