from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
def metadata_from_value(value: Any) -> dict[str, Any]:
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}
