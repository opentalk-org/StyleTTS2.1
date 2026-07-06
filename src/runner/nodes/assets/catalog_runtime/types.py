from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class CheckpointType(StrEnum):
    STYLETTS2 = "STYLETTS2"
    F0_MODEL = "F0_MODEL"
    ASR_BUNDLE = "ASR_BUNDLE"
    PLBERT = "PLBERT"


class ExtraFileType(StrEnum):
    OOD_TEXT_SET = "OOD_TEXT_SET"


@dataclass(frozen=True)
class CatalogFile:
    url: str
    name: str


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    name: str
    type_: CheckpointType
    files: tuple[CatalogFile, ...]
    metadata: dict
    is_valid: Callable
    metadata_from_path: Callable[[Path], dict[str, Any]]


@dataclass(frozen=True)
class ExtraFileSpec:
    key: str
    name: str
    type_: ExtraFileType
    url: str
    metadata: dict


@dataclass(frozen=True)
class CatalogTask:
    key: str
    run: Callable[..., dict]
