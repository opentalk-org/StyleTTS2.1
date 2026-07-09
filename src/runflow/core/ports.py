from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar


class PortMode(str, Enum):
    SINGLE = "single"
    LIST = "list"
    STREAM = "stream"


class JoinMode(str, Enum):
    ITEM = "item"
    BROADCAST = "broadcast"


@dataclass(frozen=True)
class Port:
    """A typed socket on a node.

    Subclass once per datatype — the subclass *is* the type. Two ports may be
    wired together iff they are the exact same class (see ``Graph.connect``);
    there is no subtyping or union, so a connection that type-checks is
    guaranteed to also validate at runtime.

    Instance fields are the per-use knobs (how the socket batches / joins).
    The datatype identity lives on the class as ClassVars.
    """

    mode: PortMode = PortMode.SINGLE
    join_mode: JoinMode = JoinMode.ITEM
    optional: bool = False
    default: Any = None

    # Per-type identity — set on each subclass, never a constructor arg.
    TYPE_NAME: ClassVar[str] = "ANY"
    python_type: ClassVar[type | tuple[type, ...]] = object
    color: ClassVar[str] = "#999999"
    description: ClassVar[str] = ""

    def validate(self, value: Any) -> bool:
        return isinstance(value, self.python_type)

    def to_schema(self, name: str) -> dict:
        return {
            "name": name,
            "type": self.TYPE_NAME,
            "mode": self.mode.value,
            "join_mode": self.join_mode.value,
            "optional": self.optional,
            "default": self.default,
            "description": self.description,
        }
