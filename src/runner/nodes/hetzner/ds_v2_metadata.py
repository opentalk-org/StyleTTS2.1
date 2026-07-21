from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.hetzner.ds_v2_audio import DsV2AudioOptions, audio_from_row, row_speaker_id
from runner.nodes.hetzner.ds_v2_metadata_audio import audio_metadata_from_row
from runner.nodes.hetzner.ds_v2_metadata_rows import DsV2MetadataRow, DsV2MetadataRows, load_metadata_rows
from runner.nodes.hetzner.ds_v2_selected_rows import load_selected_audio_rows
from runner.nodes.models import Audio


class HetznerDsV2SourceSettings(StrictSettings):
    host: str = Field(default="hetzner-storagebox", title="SFTP host")
    row_offset: int = Field(default=0, ge=0, title="Row offset")
    row_limit: int | None = Field(default=None, ge=1, title="Rows to import")
    import_audio: bool = Field(default=False, title="Import audio bytes")
    text_column: Literal["text_src", "text_parakeet", "text_whisper", "text_canary"] = "text_src"
    name_prefix: str = Field(default="ds_v2", title="Audio name prefix")
    download_retries: int = Field(default=3, ge=1, le=10, title="SFTP retries")


class HetznerDsV2SourceNode(Node):
    NODE_TYPE = "HetznerDsV2Source"
    DESCRIPTION = "Discover ds_v2 metadata CSVs on Hetzner and import globally selected rows. Optionally download audio bytes from each row's inferred Parquet file; otherwise emit external audio references."
    CATEGORY = "Inputs"
    SETTINGS = HetznerDsV2SourceSettings
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
        context.check_cancel()
        items = _audio_items(rows, self.settings, Path(context.cache_dir) / "hetzner")
        context.check_cancel()
        if self.settings.row_limit is None:
            self._remaining = 1
        else:
            self._remaining = max(0, self._remaining - len(items))
        return [{"audio": item} for item in items]

    async def teardown(self, context: Any) -> None:
        if self._source is not None:
            self._source.prune_cache()


def _audio_items(
    sources: list[DsV2MetadataRow],
    settings: HetznerDsV2SourceSettings,
    cache_dir: Path,
) -> list[Audio]:
    loaded = None
    if settings.import_audio:
        loaded = load_selected_audio_rows(settings.host, sources, cache_dir, settings.download_retries)
    items = []
    for position, source in enumerate(sources):
        options = DsV2AudioOptions(
            settings.host,
            source.remote_parquet_path,
            settings.text_column,
            settings.name_prefix,
        )
        speaker_id = row_speaker_id(source.metadata)
        if loaded is not None:
            row = {**source.metadata, "audio": loaded[position].audio}
            items.append(audio_from_row(row, options, source.index, speaker_id))
        else:
            items.append(
                audio_metadata_from_row(
                    source.metadata,
                    options,
                    source.index,
                    speaker_id,
                    source.remote_metadata_path,
                )
            )
    return items
