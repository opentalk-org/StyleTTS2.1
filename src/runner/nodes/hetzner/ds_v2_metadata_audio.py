from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5

from runner.nodes.hetzner.ds_v2_audio import (
    DsV2AudioOptions,
    _audio_metadata,
    _audio_name,
    _float_or_none,
    _text,
    _transcript_segments,
    speaker_id,
)
from runner.nodes.models import Audio, stable_id
from shared.audio_annotations import AudioAnnotations


def audio_metadata_from_row(
    row: dict[str, str],
    options: DsV2AudioOptions,
    row_index: int,
    voice_id: UUID | None,
    remote_metadata_path: str,
) -> Audio:
    duration = _float_or_none(row["duration"])
    if duration is None or duration <= 0:
        raise ValueError(f"ds_v2 metadata row {row_index} has invalid duration: {row['duration']!r}")
    source_key = f"{options.host}:{options.remote_parquet_path}:{row_index}"
    audio_file_id = uuid5(NAMESPACE_URL, source_key)
    score = _float_or_none(row["mos_score"])
    text = _text(row, options.text_column)
    name = _audio_name(options.name_prefix, row, row_index)
    segments = _transcript_segments(
        row, options, row_index, audio_file_id, name, duration, 0, 0, score, voice_id
    )
    metadata = _audio_metadata(
        row, options, row_index, 0, 0, duration, text
    )
    return Audio(
        audio_file_id=audio_file_id,
        name=name,
        data=None,
        sample_rate=0,
        channels=0,
        start=0.0,
        end=duration,
        annotations=AudioAnnotations(
            speaker_id=speaker_id(row),
            voice_id=voice_id,
            score=score,
            metadata={
                **metadata,
                "source_metadata_path": remote_metadata_path,
                "storage_provider": "hetzner_sftp_parquet",
            },
        ),
        id=stable_id("hetzner_ds_v2_metadata", options.remote_parquet_path, row_index),
        lineage_id=stable_id("hetzner_ds_v2_metadata_lineage", options.remote_parquet_path, row_index),
        byte_length=0,
        virtual=True,
        segments=segments,
    )
