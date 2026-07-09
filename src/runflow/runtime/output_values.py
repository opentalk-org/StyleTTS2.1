from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode


def output_values(node: Node, port_name: str, port: Port, value: Any) -> list[Any]:
    if port.mode == PortMode.STREAM and _is_stream_iterable(value):
        values = list(value)
    else:
        values = [value]

    for item in values:
        if not port.validate(item):
            raise TypeError(
                f"{node.id} returned invalid value for output port {port_name}: "
                f"expected {port.TYPE_NAME}, got {type(item).__name__}"
            )
    return values


def _is_stream_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict, Path)):
        return False
    return isinstance(value, Iterable)
