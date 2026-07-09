from __future__ import annotations

from runflow.core.ports import Port


class TypeRegistry:
    def __init__(self) -> None:
        self.types: dict[str, type[Port]] = {}

    def register(self, port_cls: type[Port]) -> type[Port]:
        if port_cls.TYPE_NAME in self.types:
            raise ValueError(f"port type already registered: {port_cls.TYPE_NAME}")
        self.types[port_cls.TYPE_NAME] = port_cls
        return port_cls

    def get(self, name: str) -> type[Port]:
        return self.types[name]

    def to_schema(self) -> dict:
        return {
            name: {
                "name": port_cls.TYPE_NAME,
                "description": port_cls.description,
                "color": port_cls.color,
            }
            for name, port_cls in self.types.items()
        }
