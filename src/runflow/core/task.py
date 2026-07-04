from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ArtifactRef:
    artifact_id: str
    dtype: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Packet:
    node_id: str
    port: str
    dtype: str
    value: Any
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    node_id: str
    inputs: dict[str, Any]
    input_packets: dict[str, Packet]
    lineage_id: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def lineage_from_value(value: Any, inherited: str | None = None) -> str:
    lineage = getattr(value, "lineage_id", None)
    if lineage:
        return str(lineage)

    value_id = getattr(value, "id", None)
    if value_id:
        return str(value_id)

    if isinstance(value, Path):
        return f"path:{value.resolve()}"

    if inherited:
        return inherited

    return f"lineage:{uuid4()}"


def metadata_from_value(value: Any) -> dict[str, Any]:
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}
