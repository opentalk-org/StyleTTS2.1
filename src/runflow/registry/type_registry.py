from __future__ import annotations

from runflow.core.types import DataType, UnionDataType


class TypeRegistry:
    def __init__(self) -> None:
        self.types: dict[str, DataType | UnionDataType] = {}

    def register(self, dtype: DataType | UnionDataType) -> DataType | UnionDataType:
        if dtype.name in self.types:
            raise ValueError(f"datatype already registered: {dtype.name}")
        self.types[dtype.name] = dtype
        return dtype

    def get(self, name: str) -> DataType | UnionDataType:
        return self.types[name]

    def to_schema(self) -> dict:
        return {
            name: {
                "name": dtype.name,
                "description": getattr(dtype, "description", ""),
                "color": getattr(dtype, "color", "#999999"),
                "members": [m.name for m in getattr(dtype, "members", ())],
            }
            for name, dtype in self.types.items()
        }
