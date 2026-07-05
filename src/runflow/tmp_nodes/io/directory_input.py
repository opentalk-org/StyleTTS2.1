from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import NodeSettings
from runflow.tmp_nodes.audio.datatypes import PATH


class DirectoryInputSettings(NodeSettings):
    directory: Path
    patterns: list[str] = Field(default_factory=lambda: ["*.wav", "*.mp3", "*.flac"])
    repeat_count: int = 1
    sleep_sec: float = 0.0


class DirectoryInputNode(Node):
    NODE_TYPE = "DirectoryInput"
    CATEGORY = "IO"
    SETTINGS = DirectoryInputSettings

    INPUTS = {}
    OUTPUTS = {
        "paths": Port("paths", PATH, mode=PortMode.STREAM, description="Matched filesystem paths"),
    }

    def list_items(self) -> list[Path]:
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
        for index, path in enumerate(context.current_window_items):
            if self.settings.sleep_sec > 0:
                await asyncio.sleep(self.settings.sleep_sec)
            source = Path(path)
            suffix = source.suffix or ".dat"
            out_path = out_dir / f"{source.stem}_{context.window_index:04d}_{index:04d}{suffix}"
            if source.exists():
                shutil.copyfile(source, out_path)
            else:
                out_path.write_text(f"placeholder source for {source}\n", encoding="utf-8")
            outputs.append({"paths": out_path})
        return outputs
