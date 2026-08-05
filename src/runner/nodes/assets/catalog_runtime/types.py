from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class CheckpointType(StrEnum):
    STYLETTS2 = "styletts2"
    F0_MODEL = "f0_model"
    ASR_BUNDLE = "asr_bundle"
    PLBERT = "plbert"
    PIPER = "piper"


class ExtraFileType(StrEnum):
    OOD_TEXT_SET = "ood_text_set"


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
