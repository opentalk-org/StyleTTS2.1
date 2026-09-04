from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from runner.nodes.models import Audio
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.clickhouse.models import AudioFileRecord
from shared.db.audio.schemas import AudioCreate
from shared.db.datasets import crud as dataset_crud


def persist_split_records(
    audios: list[Audio],
    payloads: list[AudioCreate],
    segment_payloads: list[list[dict]],
    target_dataset_id: UUID | None,
    source_dataset_id: UUID | None,
    replace_all: bool,
    delete_source: bool,
) -> tuple[list[AudioFileRecord], dict[UUID, list[dict]]]:
    with database_session() as session:
        items = audio_crud.bulk_create_audio_files(session, payloads)
        segments_by_id = audio_crud.bulk_replace_audio_segments(
            {
                item.id: segments
                for item, segments in zip(items, segment_payloads, strict=True)
            },
        )
        completed_source_ids = completed_replace_source_ids(
            session,
            audios,
            replace_all,
        )
        dataset_ids = [
            dataset_id
            for dataset_id in dict.fromkeys([target_dataset_id, source_dataset_id])
            if dataset_id is not None
        ]
        for dataset_id in dataset_ids:
            dataset_crud.bulk_add_audio_files_to_dataset(
                dataset_id,
                [item.id for item in items],
            )
        if source_dataset_id is not None and completed_source_ids:
            dataset_crud.bulk_remove_audio_files_from_dataset(
                source_dataset_id,
                completed_source_ids,
            )
        if delete_source and completed_source_ids:
            audio_crud.bulk_delete_audio_files(
                session,
                completed_source_ids,
            )
    return items, segments_by_id


def completed_replace_source_ids(
    session: Session,
    audios: list[Audio],
    replace_all: bool,
) -> list[UUID]:
    if not replace_all:
        return []
    operation_by_source: dict[UUID, str] = {}
    for audio in audios:
        source_id = UUID(str(audio.metadata["source_audio_id"]))
        operation_id = str(audio.metadata["split_operation_id"])
        previous = operation_by_source.setdefault(source_id, operation_id)
        if previous != operation_id:
            raise ValueError(f"mixed split operations: {source_id}")
    source_ids = list(operation_by_source)
    operation_ids = list(dict.fromkeys(operation_by_source.values()))
    groups: dict[UUID, list[tuple[int, int]]] = {
        source_id: [] for source_id in source_ids
    }
    for metadata in audio_crud.list_split_metadata(source_ids, operation_ids):
        source_id = UUID(str(metadata["source_audio_id"]))
        if str(metadata["split_operation_id"]) != operation_by_source[source_id]:
            continue
        groups[source_id].append(
            (int(metadata["group_index"]), int(metadata["group_count"]))
        )
    completed = []
    for source_id, group_items in groups.items():
        counts = {count for _index, count in group_items}
        if len(counts) != 1:
            raise ValueError(f"inconsistent split group_count: {source_id}")
        count = counts.pop()
        indices = [index for index, _count in group_items]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate split group_index: {source_id}")
        expected = set(range(count))
        if not set(indices).issubset(expected):
            raise ValueError(f"invalid split group_index: {source_id}")
        if set(indices) == expected:
            completed.append(source_id)
    return completed
