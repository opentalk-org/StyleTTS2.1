from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import UUID

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.hetzner.ds_v2_audio import DsV2AudioOptions, speaker_name
from runner.nodes.hetzner.ds_v2_metadata_audio import audio_metadata_from_row
from runner.nodes.hetzner.ds_v2_metadata_rows import DsV2MetadataRow, DsV2MetadataRows, load_metadata_rows
from runner.nodes.models import Audio
from shared.db import database_session
from shared.db.voices.models import Voice


class HetznerDsV2MetadataSourceSettings(StrictSettings):
    host: str = Field(default="hetzner-storagebox", title="SFTP host")
    row_offset: int = Field(default=0, ge=0, title="Row offset")
    row_limit: int | None = Field(default=None, ge=1, title="Rows to import")
    text_column: Literal["text_src", "text_parakeet", "text_whisper", "text_canary"] = "text_src"
    name_prefix: str = Field(default="ds_v2", title="Audio name prefix")
    download_retries: int = Field(default=3, ge=1, le=10, title="SFTP retries")
    create_voices: bool = Field(default=True, title="Create voices")


class HetznerDsV2MetadataSourceNode(Node):
    NODE_TYPE = "HetznerDsV2MetadataSource"
    DESCRIPTION = "Discover ds_v2 metadata CSVs on Hetzner and import their rows as virtual audio references without downloading Parquet audio bytes. Offset and row limit apply across all discovered CSVs."
    CATEGORY = "Inputs"
    SETTINGS = HetznerDsV2MetadataSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 512

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._source: DsV2MetadataRows | None = None
        self._rows: Iterator[DsV2MetadataRow] | None = None
        self._remaining: int | None = None

    def remaining_items(self, context: Any) -> int:
        if self._remaining is not None:
            return self._remaining
        return self.settings.row_limit or 1

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        if self._rows is None:
            self._source = load_metadata_rows(
                self.settings.host,
                Path(context.cache_dir) / "hetzner",
                self.settings.download_retries,
                self.settings.row_offset,
                self.settings.row_limit,
            )
            self._rows = iter(self._source)
            self._remaining = self.settings.row_limit or 1
        rows = list(islice(self._rows, self.runtime.queue_max_size))
        if not rows:
            self._remaining = 0
            return []
        voice_ids = _voice_ids(rows, self.settings.create_voices)
        items = [_audio_from_source(source, self.settings, voice_ids) for source in rows]
        if self.settings.row_limit is None:
            self._remaining = 1
        else:
            self._remaining = max(0, self._remaining - len(items))
        return [{"audio": item} for item in items]

    async def teardown(self, context: Any) -> None:
        if self._source is not None:
            self._source.prune_cache()


def _audio_from_source(
    source: DsV2MetadataRow,
    settings: HetznerDsV2MetadataSourceSettings,
    voice_ids: dict[str, UUID],
) -> Audio:
    options = DsV2AudioOptions(
        settings.host,
        source.remote_parquet_path,
        settings.text_column,
        settings.name_prefix,
    )
    return audio_metadata_from_row(
        source.metadata,
        options,
        source.index,
        voice_ids.get(speaker_name(source.metadata)),
        source.remote_metadata_path,
    )


def _voice_ids(rows: list[DsV2MetadataRow], enabled: bool) -> dict[str, UUID]:
    if not enabled:
        return {}
    names = sorted({name for source in rows if (name := speaker_name(source.metadata))})
    if not names:
        return {}
    with database_session() as session:
        session.execute(insert(Voice).values([{"name": name} for name in names]).on_conflict_do_nothing(index_elements=["name"]))
        session.commit()
        voices = session.execute(select(Voice).where(Voice.name.in_(names))).scalars().all()
        return {voice.name: voice.id for voice in voices}
