from collections.abc import Sequence
from uuid import UUID

from shared.db.clickhouse import clickhouse_client, delete_rows
from shared.db.waveforms.clickhouse.models import AudioWaveformRecord


def replace_waveform(item: AudioWaveformRecord) -> AudioWaveformRecord:
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
        WHERE audio_file_id = {audio_file_id:UUID}
        """,
        parameters={"audio_file_id": audio_file_id},
    )
    rows = list(result.named_results())
    if not rows:
        raise KeyError(f"Waveform not found: {audio_file_id}")
    return AudioWaveformRecord.model_validate(rows[0])


def delete_waveforms(audio_file_ids: Sequence[UUID]) -> None:
    if not audio_file_ids:
        return
    delete_rows(
        clickhouse_client(),
        "audio_waveforms",
        "audio_file_id IN {ids:Array(UUID)}",
        {"ids": list(audio_file_ids)},
    )
