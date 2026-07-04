from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runflow.core.task import Packet


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index: list[dict[str, Any]] = []

    def register_packet(self, packet: Packet) -> None:
        path = getattr(packet.value, "path", None)
        self.index.append(
            {
                "node_id": packet.node_id,
                "port": packet.port,
                "dtype": packet.dtype,
                "lineage_id": packet.lineage_id,
                "path": str(path) if path else None,
                "metadata": packet.metadata,
            }
        )

    def write_index(self, filename: str = "artifact_index.json") -> Path:
        out = self.root / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.index, indent=2, ensure_ascii=False), encoding="utf-8")
        return out
