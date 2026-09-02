from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from shared.db.audio.clickhouse.segments import list_audio_segments_bulk
from shared.db.audio.clickhouse.files import get_audio_files
from shared.db.audio.clickhouse.models import AudioFileRecord
from shared.db.clickhouse import clickhouse_client
from shared.db.datasets.schemas import DatasetDurationBins, DatasetTrainingAudio


@dataclass(frozen=True)
class TtsReferenceCandidate:
    audio: AudioFileRecord
    segments: list[dict[str, Any]]


def iter_dataset_training_audio(
    dataset_id: UUID,
    descending: bool = False,
    duration_above: float | None = None,
    duration_at_most: float | None = None,
    excluded_audio_ids: set[UUID] | None = None,
    audio_id_after: UUID | None = None,
    audio_id_at_most: UUID | None = None,
) -> Iterator[DatasetTrainingAudio]:
    direction = "DESC" if descending else "ASC"
    after_operator = "<" if descending else ">"
    cursor = audio_id_after
    while True:
        filters = ["membership.dataset_id = {dataset_id:UUID}", "audio.virtual = false"]
        parameters: dict[str, object] = {"dataset_id": dataset_id, "limit": 2_000}
        if duration_above is not None:
            filters.append("audio.duration > {duration_above:Float64}")
            parameters["duration_above"] = duration_above
        if duration_at_most is not None:
            filters.append("audio.duration <= {duration_at_most:Float64}")
            parameters["duration_at_most"] = duration_at_most
        if excluded_audio_ids:
            filters.append("audio.id NOT IN {excluded:Array(UUID)}")
            parameters["excluded"] = list(excluded_audio_ids)
        if cursor is not None:
            filters.append(f"audio.id {after_operator} {{cursor:UUID}}")
            parameters["cursor"] = cursor
        if audio_id_at_most is not None:
            filters.append("audio.id <= {at_most:UUID}")
            parameters["at_most"] = audio_id_at_most
        result = clickhouse_client().query(
            f"""
            SELECT audio.id, audio.duration, audio.language, audio.metadata
            FROM ({_LATEST_TRAINING_AUDIO}) AS audio
            INNER JOIN dataset_audio_files AS membership FINAL ON membership.audio_file_id = audio.id
            WHERE {" AND ".join(filters)}
            ORDER BY audio.id {direction}
            LIMIT {{limit:UInt32}}
            """,
            parameters=parameters,
        )
        rows = result.result_rows
        if not rows:
            return
        segments = list_audio_segments_bulk([row[0] for row in rows])
        for row in rows:
            payloads = [segment.as_payload() for segment in segments[row[0]]]
            speakers = {segment.speaker_id for segment in segments[row[0]]}
            yield DatasetTrainingAudio(
                row[0],
                row[1],
                speakers.pop() if len(speakers) == 1 else None,
                row[2],
                payloads,
                row[3],
            )
        cursor = rows[-1][0]


def count_dataset_training_audio(dataset_id: UUID) -> int:
    result = clickhouse_client().query(
        f"""
        SELECT count()
        FROM ({_LATEST_TRAINING_AUDIO}) AS audio
        INNER JOIN dataset_audio_files AS membership FINAL ON membership.audio_file_id = audio.id
        WHERE membership.dataset_id = {{dataset_id:UUID}} AND audio.virtual = false
        """,
        parameters={"dataset_id": dataset_id},
    )
    return int(result.result_rows[0][0])


def list_dataset_metadata_values(dataset_id: UUID, key: str) -> set[str]:
    result = clickhouse_client().query(
        f"""
        SELECT DISTINCT JSONExtractString(audio.metadata, {{key:String}})
        FROM ({_LATEST_TRAINING_AUDIO}) AS audio
        INNER JOIN dataset_audio_files AS membership FINAL ON membership.audio_file_id = audio.id
        WHERE membership.dataset_id = {{dataset_id:UUID}}
          AND JSONHas(audio.metadata, {{key:String}})
        """,
        parameters={"dataset_id": dataset_id, "key": key},
    )
    return {str(row[0]) for row in result.result_rows}


def list_tts_reference_candidates(
    dataset_ids: Sequence[UUID], streams: Sequence[str]
) -> list[TtsReferenceCandidate]:
    result = clickhouse_client().query(
        """
        SELECT DISTINCT audio_file_id
        FROM dataset_audio_files FINAL
        WHERE dataset_id IN {dataset_ids:Array(UUID)}
        """,
        parameters={"dataset_ids": list(dataset_ids)},
    )
    ids = [row[0] for row in result.result_rows]
    files = get_audio_files(ids)
    segments = list_audio_segments_bulk(ids)
    return [
        TtsReferenceCandidate(
            item,
            [segment.as_payload() for segment in segments[item.id]],
        )
        for item in files
        if str(item.metadata["stream"]) in streams
    ]


def dataset_training_duration_totals(
    dataset_id: UUID, upper_bounds: tuple[float, ...], excluded_audio_ids: set[UUID]
) -> DatasetDurationBins:
    rows = list(
        iter_dataset_training_audio(
            dataset_id,
            duration_above=0,
            duration_at_most=upper_bounds[-1],
            excluded_audio_ids=excluded_audio_ids,
        )
    )
    totals = [0.0] * len(upper_bounds)
    for row in rows:
        index = next(
            index for index, bound in enumerate(upper_bounds) if row.duration <= bound
        )
        totals[index] += row.duration
    return DatasetDurationBins(tuple(totals), len(rows))


def dataset_training_minimum_duration(
    dataset_id: UUID, max_duration: float, excluded_audio_ids: set[UUID]
) -> float:
    rows = iter_dataset_training_audio(
        dataset_id,
        duration_above=0,
        duration_at_most=max_duration,
        excluded_audio_ids=excluded_audio_ids,
    )
    try:
        return min(row.duration for row in rows)
    except ValueError as error:
        raise ValueError(
            "training dataset has no audio within the duration limit"
        ) from error


_LATEST_TRAINING_AUDIO = """
SELECT id, latest.1 AS duration, latest.2 AS language, latest.3 AS metadata, latest.4 AS virtual
FROM (SELECT id, argMax(tuple(duration, language, metadata, virtual), updated_at) AS latest
      FROM audio_files GROUP BY id)
"""
