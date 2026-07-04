from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class TypeLike(Protocol):
    name: str

    def validate(self, value: Any) -> bool:
        ...


@dataclass(frozen=True)
class DataType:
    """A runtime/UI datatype for a typed graph socket."""

    name: str
    python_type: type | tuple[type, ...] = object
    description: str = ""
    color: str = "#999999"
    validator: Callable[[Any], bool] | None = None

    def validate(self, value: Any) -> bool:
        if self.validator is not None:
            return self.validator(value)
        return isinstance(value, self.python_type)


@dataclass(frozen=True)
class UnionDataType:
    """A datatype that accepts one of several concrete datatypes."""

    name: str
    members: tuple[DataType, ...]
    description: str = ""
    color: str = "#CCCCCC"

    def validate(self, value: Any) -> bool:
        return any(member.validate(value) for member in self.members)


def dtype_accepts(target: DataType | UnionDataType, source: DataType | UnionDataType) -> bool:
    """Return True when an output dtype can be connected to an input dtype."""

    if isinstance(target, UnionDataType):
        if isinstance(source, UnionDataType):
            return all(any(m.name == sm.name for m in target.members) for sm in source.members)
        return any(member.name == source.name for member in target.members)

    if isinstance(source, UnionDataType):
        return all(dtype_accepts(target, member) for member in source.members)

    return target.name == source.name
