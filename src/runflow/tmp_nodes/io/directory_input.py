from __future__ import annotations

from pathlib import Path

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.tmp_nodes.audio.datatypes import PATH


class DirectoryInputNode(Node):
    NODE_TYPE = "DirectoryInput"
    CATEGORY = "IO"

    INPUTS = {}
    OUTPUTS = {
        "paths": Port("paths", PATH, mode=PortMode.STREAM, description="Audio file paths"),
    }

    def list_items(self) -> list[Path]:
        directory = Path(self.params.get("directory", "input_audio"))
        patterns = self.params.get("patterns", ["*.wav", "*.mp3", "*.flac", "*.m4a"])
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
        return unique

    def list_paths(self) -> list[Path]:
        return self.list_items()

    def execute(self, batch, context):
        return [{"paths": path} for path in context.current_window_items]
