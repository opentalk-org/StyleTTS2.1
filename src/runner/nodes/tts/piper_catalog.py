from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict


PIPER_VOICES_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_CATALOG_URL = f"{PIPER_VOICES_BASE_URL}/voices.json"


class PiperLanguage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    family: str
    region: str
    name_native: str
    name_english: str
    country_english: str


class PiperFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    size_bytes: int
    md5_digest: str


class PiperVoiceEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    name: str
    language: PiperLanguage
    quality: str
    num_speakers: int
    speaker_id_map: dict[str, int]
    files: dict[str, PiperFile]
    aliases: list[str]

    @property
    def voice_id(self) -> str:
        return self.key

    @property
    def model_path(self) -> str:
        paths = [path for path in self.files if path.endswith(".onnx")]
        if len(paths) != 1:
            raise ValueError(f"piper_voice_model_count:{self.key}:{len(paths)}")
        return paths[0]

    @property
    def config_path(self) -> str:
        paths = [path for path in self.files if path.endswith(".onnx.json")]
        if len(paths) != 1:
            raise ValueError(f"piper_voice_config_count:{self.key}:{len(paths)}")
        return paths[0]


PiperCatalog = tuple[PiperVoiceEntry, ...]


@dataclass(frozen=True)
class PiperSelection:
    voice_ids: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    locales: tuple[str, ...] = ()
    qualities: tuple[str, ...] = ()
    count: int | None = None
    seed: int = 0


def parse_catalog(payload: dict[str, Any]) -> PiperCatalog:
    voices = tuple(PiperVoiceEntry.model_validate(item) for item in payload.values())
    if len({voice.key for voice in voices}) != len(voices):
        raise ValueError("piper_catalog_duplicate_voice_id")
    for voice in voices:
        voice.model_path
        voice.config_path
    return tuple(sorted(voices, key=lambda voice: voice.key))


def fetch_piper_catalog() -> PiperCatalog:
    request = Request(PIPER_CATALOG_URL, headers={"User-Agent": "Runflow-Studio/2.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("piper_catalog_invalid_root")
    return parse_catalog(payload)


def select_voices(catalog: PiperCatalog, selection: PiperSelection) -> PiperCatalog:
    by_id = {voice.key: voice for voice in catalog}
    if selection.voice_ids:
        missing = [
            voice_id for voice_id in selection.voice_ids if voice_id not in by_id
        ]
        if missing:
            raise ValueError(f"piper_unknown_voice_ids:{','.join(missing)}")
        return tuple(by_id[voice_id] for voice_id in selection.voice_ids)

    candidates = tuple(voice for voice in catalog if _matches(voice, selection))
    if not candidates:
        raise ValueError("piper_voice_selection_empty")
    if selection.count is None:
        return candidates
    take = min(selection.count, len(candidates))
    return tuple(random.Random(selection.seed).sample(list(candidates), take))


def _matches(voice: PiperVoiceEntry, selection: PiperSelection) -> bool:
    language_matches = (
        not selection.languages or voice.language.family in selection.languages
    )
    locale_matches = not selection.locales or voice.language.code in selection.locales
    quality_matches = not selection.qualities or voice.quality in selection.qualities
    return language_matches and locale_matches and quality_matches
