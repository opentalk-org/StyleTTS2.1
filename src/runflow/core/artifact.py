from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class StoredArtifact:
    dtype: str
    path: Path | None = None
    value: Any | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    lineage_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
