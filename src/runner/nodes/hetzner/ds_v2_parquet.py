from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.hetzner.ds_v2_audio import DsV2AudioOptions, audio_from_row, speaker_name
from runner.nodes.hetzner.ds_v2_rows import load_rows
from runner.nodes.models import Audio
from shared.db import database_session
from shared.db.voices.models import Voice


DEFAULT_PARQUET_PATH = "/home/ds_v2/000f72c2-caa7-4958-b8e8-0e7668bb9bb6_20260512T173847038808Z_processed.parquet"


class HetznerDsV2ParquetAudioSourceSettings(StrictSettings):
    host: str = Field(default="hetzner-storagebox", title="SFTP host")
    remote_parquet_path: str = Field(default=DEFAULT_PARQUET_PATH, title="Remote parquet path")
    row_offset: int = Field(default=0, ge=0, title="Row offset")
    row_limit: int = Field(default=1, ge=1, title="Rows to import")
    text_column: Literal["text_src", "text_parakeet", "text_whisper", "text_canary"] = Field(
        default="text_src",
        title="Transcript column",
    )
    name_prefix: str = Field(default="ds_v2", title="Audio name prefix")
    download_retries: int = Field(default=3, ge=1, le=10, title="SFTP retries")
    create_voices: bool = Field(default=True, title="Create voices")


class HetznerDsV2ParquetAudioSourceNode(Node):
    NODE_TYPE = "HetznerDsV2ParquetAudioSource"
    DESCRIPTION = "Import ds_v2 audio from a cached Hetzner Parquet file and metadata from its exact cached /home/ds_v2_metadata CSV pair. The pair is validated row-for-row before emitting WAV audio and transcript segments; missing or mismatched metadata fails the run. The cache retains the four most recently used ds_v2 pairs."
    CATEGORY = "Inputs"
    SETTINGS = HetznerDsV2ParquetAudioSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._items: list[Audio] | None = None
        self._cursor = 0

    def remaining_items(self, context: Any) -> int:
        if self._items is None:
            return self.settings.row_limit
        return len(self._items) - self._cursor

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        if self._items is None:
            self._items = _load_audio_items(self.settings, context)
        end = self._cursor + self.runtime.queue_max_size
        items = self._items[self._cursor:end]
        self._cursor += len(items)
        return [{"audio": item} for item in items]


def _load_audio_items(settings: HetznerDsV2ParquetAudioSourceSettings, context: Any) -> list[Audio]:
    source_rows = load_rows(
        host=settings.host,
        remote_parquet_path=settings.remote_parquet_path,
        cache_dir=Path(context.cache_dir) / "hetzner",
        retries=settings.download_retries,
        row_offset=settings.row_offset,
        row_limit=settings.row_limit,
    )
    metadata_rows = [source_row.metadata for source_row in source_rows]
    voice_ids = _voice_ids_for_rows(settings, metadata_rows)
    options = DsV2AudioOptions(
        host=settings.host,
        remote_parquet_path=settings.remote_parquet_path,
        text_column=settings.text_column,
        name_prefix=settings.name_prefix,
    )
    return [
        audio_from_row(
            {**source_row.metadata, "audio": source_row.audio},
            options,
            source_row.index,
            voice_ids.get(speaker_name(source_row.metadata)),
        )
        for source_row in source_rows
    ]


def _voice_ids_for_rows(
    settings: HetznerDsV2ParquetAudioSourceSettings,
    rows: list[dict[str, str]],
) -> dict[str, UUID]:
    if not settings.create_voices:
        return {}
    names = sorted({speaker for row in rows if (speaker := speaker_name(row))})
    if not names:
        return {}
    with database_session() as session:
        session.execute(
            insert(Voice)
            .values([{"name": name} for name in names])
            .on_conflict_do_nothing(index_elements=["name"])
        )
        session.commit()
        voices = session.execute(select(Voice).where(Voice.name.in_(names))).scalars().all()
        return {voice.name: voice.id for voice in voices}
