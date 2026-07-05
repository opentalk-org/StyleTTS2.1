from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.tmp_nodes.audio.datatypes import PATH


class DirectoryInputSettings(StrictSettings):
    directory: Path
    patterns: list[str] = Field(default_factory=lambda: ["*.wav", "*.mp3", "*.flac"])
    repeat_count: int = 1
    sleep_sec: float = 0.0


class DirectoryInputNode(Node):
    NODE_TYPE = "DirectoryInput"
    CATEGORY = "IO"
    SETTINGS = DirectoryInputSettings
    IS_INPUT = True

    INPUTS = {}
    OUTPUTS = {
        "paths": Port("paths", PATH, mode=PortMode.STREAM, description="Matched filesystem paths"),
    }

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._items = self._load_items()
        self._cursor = 0

    def remaining_items(self, context) -> int:
        return len(self._items) - self._cursor

    def _load_items(self) -> list[Path]:
        directory = Path(self.params["directory"])
        patterns = self.params["patterns"]
        repeat_count = max(1, int(self.params["repeat_count"]))
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(sorted(directory.rglob(pattern)))
        # Deduplicate while preserving order.
        seen = set()
        unique = []
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)
        return [path for path in unique for _ in range(repeat_count)]

    async def execute(self, batch, context):
        outputs = []
        out_dir = context.node_dir(self.id)
        start = self._cursor
        window_size = self.runtime.window_size
        if window_size is None:
            raise RuntimeError(f"Input node {self.id} requires runtime.window_size")
        items = self._items[start:start + window_size]
        for index, path in enumerate(items):
            if self.settings.sleep_sec > 0:
                await asyncio.sleep(self.settings.sleep_sec)
            source = Path(path)
            suffix = source.suffix or ".dat"
            out_path = out_dir / f"{source.stem}_{context.window_index:04d}_{start + index:04d}{suffix}"
            if source.exists():
                shutil.copyfile(source, out_path)
            else:
                out_path.write_text(f"placeholder source for {source}\n", encoding="utf-8")
            outputs.append({"paths": out_path})
        self._cursor += len(items)
        return outputs
