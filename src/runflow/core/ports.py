from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from runflow.core.types import DataType, UnionDataType


class PortMode(str, Enum):
    SINGLE = "single"
    LIST = "list"
    STREAM = "stream"


@dataclass(frozen=True)
class Port:
    name: str
    dtype: DataType | UnionDataType
    mode: PortMode = PortMode.SINGLE
    optional: bool = False
    default: Any = None
    description: str = ""

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "type": self.dtype.name,
            "mode": self.mode.value,
            "optional": self.optional,
            "default": self.default,
            "description": self.description,
        }
