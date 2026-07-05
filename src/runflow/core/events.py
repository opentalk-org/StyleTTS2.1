from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class RunEvent:
    kind: str
    run_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    message: str = ""
    node_id: str | None = None
    port: str | None = None
    target_node_id: str | None = None
    target_port: str | None = None
    window_index: int | None = None
    worker_index: int | None = None
    batch_index: int | None = None
    batch_size: int | None = None
    lineage_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
