from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

from shared.db.clickhouse import clickhouse_client, delete_rows
from shared.db.waveforms.clickhouse.models import AudioWaveformRecord


def replace_waveform(item: AudioWaveformRecord) -> AudioWaveformRecord:
    try:
        current = get_waveform(item.audio_file_id)
    except KeyError:
        current = None
    if current is not None and item.updated_at <= current.updated_at:
        item = item.model_copy(
            update={"updated_at": current.updated_at + timedelta(microseconds=1)}
        )
    clickhouse_client().insert(
        "audio_waveforms",
        [
            [
                item.audio_file_id,
                item.updated_at,
                item.pack_id,
                item.byte_offset,
                item.byte_length,
                item.duration,
                item.sample_rate,
                item.points_per_second,
                item.point_count,
            ]
        ],
        column_names=[
            "audio_file_id",
            "updated_at",
            "pack_id",
            "byte_offset",
            "byte_length",
            "duration",
            "sample_rate",
            "points_per_second",
            "point_count",
        ],
    )
    return get_waveform(item.audio_file_id)


def get_waveform(audio_file_id: UUID) -> AudioWaveformRecord:
    rows = get_waveforms([audio_file_id])
    if not rows:
        raise KeyError(f"Waveform not found: {audio_file_id}")
    return rows[0]


def get_waveforms(audio_file_ids: Sequence[UUID]) -> list[AudioWaveformRecord]:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return []
    result = clickhouse_client().query(
        """
        SELECT
            audio_file_id,
            updated_at,
            pack_id,
            byte_offset,
            byte_length,
            duration,
            sample_rate,
            points_per_second,
            point_count
        FROM audio_waveforms FINAL
        WHERE audio_file_id IN {ids:Array(UUID)}
        """,
        parameters={"ids": ids},
    )
    return [AudioWaveformRecord.model_validate(row) for row in result.named_results()]


def waveform_exists(audio_file_id: UUID) -> bool:
    result = clickhouse_client().query(
        """
        SELECT 1
        FROM audio_waveforms FINAL
        WHERE audio_file_id = {audio_file_id:UUID}
        LIMIT 1
        """,
        parameters={"audio_file_id": audio_file_id},
    )
    return bool(result.result_rows)


def delete_waveforms(audio_file_ids: Sequence[UUID]) -> None:
    if not audio_file_ids:
        return
    delete_rows(
        clickhouse_client(),
        "audio_waveforms",
        "audio_file_id IN {ids:Array(UUID)}",
        {"ids": list(audio_file_ids)},
    )
