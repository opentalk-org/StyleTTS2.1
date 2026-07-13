import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Text, cast, desc, func, or_, select
from sqlalchemy.orm import Session

from shared.db.audio.delete_crud import bulk_delete_audio_files, delete_audio_file
from shared.db.audio.models import AudioFile
from shared.db.audio.pack_crud import (
    bulk_create_packed_audio_files,
    bulk_read_packed_audio_files,
    bulk_read_packed_audio_parts,
    bulk_update_packed_audio_files,
    create_packed_audio_file,
    read_packed_audio_file,
    read_packed_audio_part,
)
from shared.db.audio.pack_prune import prune_fragmented_audio_packs
from shared.db.audio.pack_store import AudioPackConfig, ObjectStore
from shared.db.audio.schemas import AudioBucketLocation, AudioCreate, AudioPartRead, AudioUpdate
from shared.db.audio.scores_crud import bulk_update_audio_scores
from shared.db.audio.rows_crud import get_audio_files_bulk
from shared.db.audio.references_crud import count_audio_file_references, list_audio_file_references_page
from shared.db.audio.segments_crud import (
    create_segment,
    delete_segment,
    list_audio_segments,
    list_audio_segments_bulk,
    replace_audio_segments,
    bulk_replace_audio_segments,
    update_segment,
    update_segment_phonemes,
    update_segment_text,
)
from shared.db.common import many
from shared.db.settings import crud as settings_crud
from shared.db.datasets.models import Dataset
from shared.storage import S3ObjectStore


def list_audio_files(session: Session) -> Sequence[AudioFile]:
    return many(session, AudioFile)


def list_audio_files_by_run(session: Session, run_id: str) -> Sequence[AudioFile]:
    statement = (
        select(AudioFile)
        .where(AudioFile.metadata_["run_id"].astext == run_id)
        .order_by(AudioFile.updated_at.asc())
    )
    return session.execute(statement).unique().scalars().all()


def search_audio_files(
    session: Session,
    query: str,
    dataset: str,
    sort: str,
    limit: int,
    offset: int,
) -> tuple[Sequence[AudioFile], int]:
    statement = select(AudioFile)
    count_statement = select(func.count()).select_from(AudioFile)
    for item in _audio_filters(query, dataset):
        statement = statement.where(item)
        count_statement = count_statement.where(item)
    statement = statement.order_by(_audio_sort(sort)).limit(limit).offset(offset)
    rows = session.execute(statement).unique().scalars().all()
    total = session.execute(count_statement).scalar_one()
    return rows, total


def search_audio_file_ids(session: Session, query: str, dataset: str) -> list[uuid.UUID]:
    statement = select(AudioFile.id)
    for item in _audio_filters(query, dataset):
        statement = statement.where(item)
    return list(session.execute(statement).scalars().all())


def get_audio_file(session: Session, audio_file_id: uuid.UUID) -> AudioFile:
    return get_audio_files_bulk(session, [audio_file_id])[audio_file_id]


def _audio_filters(query: str, dataset: str) -> list[Any]:
    filters = []
    if query:
        pattern = f"%{query}%"
        filters.append(or_(AudioFile.name.ilike(pattern), cast(AudioFile.metadata_, Text).ilike(pattern)))
    if dataset == "unassigned":
        filters.append(~AudioFile.datasets.any())
    elif dataset != "all":
        dataset_id = uuid.UUID(dataset)
        filters.append(AudioFile.datasets.any(Dataset.id == dataset_id))
    return filters


def _audio_sort(sort: str):
    if sort == "name":
        return AudioFile.name
    if sort == "duration":
        return desc(AudioFile.duration)
    if sort == "segments":
        return desc(func.jsonb_array_length(AudioFile.segments))
    if sort == "speaker":
        return AudioFile.name
    return desc(AudioFile.updated_at)


def create_audio_file(
    session: Session,
    payload: AudioCreate,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    resolved_store = _object_store(session, store)
    return create_packed_audio_file(session, resolved_store, payload, config)


def bulk_create_audio_files(
    session: Session,
    payloads: Sequence[AudioCreate],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
) -> list[AudioFile]:
    resolved_store = _object_store(session, store)
    return bulk_create_packed_audio_files(
        session,
        resolved_store,
        payloads,
        config,
        commit=commit,
    )


def read_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_file(session, _object_store(session, store), audio_file_id)


def read_audio_part(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioPartRead,
    store: ObjectStore | None = None,
) -> bytes:
    return read_packed_audio_part(session, _object_store(session, store), audio_file_id, payload)


def bulk_read_audio_files(
    session: Session,
    audio_file_ids: Iterable[uuid.UUID],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_files(session, _object_store(session, store), audio_file_ids)


def bulk_read_audio_parts(
    session: Session,
    requests: dict[uuid.UUID, AudioPartRead],
    store: ObjectStore | None = None,
) -> dict[uuid.UUID, bytes]:
    return bulk_read_packed_audio_parts(session, _object_store(session, store), requests)


def audio_bucket_locations(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
) -> list[AudioBucketLocation]:
    """Return the bucket file id and byte size for each audio id.

    Lets callers group and size work by bucket (e.g. streaming a training set
    bucket-by-bucket) without touching pack offsets. Order follows the request."""
    statement = select(
        AudioFile.id, AudioFile.bucket_file_id, AudioFile.byte_length
    ).where(AudioFile.id.in_(audio_file_ids))
    rows = {
        audio_id: (bucket_file_id, byte_length)
        for audio_id, bucket_file_id, byte_length in session.execute(statement)
    }
    return [
        AudioBucketLocation(
            audio_file_id=audio_file_id,
            bucket_file_id=rows[audio_file_id][0],
            byte_length=rows[audio_file_id][1],
        )
        for audio_file_id in audio_file_ids
    ]


def update_audio_file(
    session: Session,
    audio_file_id: uuid.UUID,
    payload: AudioUpdate,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> AudioFile:
    return bulk_update_audio_files(
        session,
        {audio_file_id: payload},
        store=store,
        config=config,
    )[audio_file_id]


def bulk_update_audio_files(
    session: Session,
    payloads: dict[uuid.UUID, AudioUpdate],
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> dict[uuid.UUID, AudioFile]:
    binary_payloads = {
        audio_file_id: payload
        for audio_file_id, payload in payloads.items()
        if payload.wav_bytes is not None
    }
    metadata_payloads = {
        audio_file_id: payload
        for audio_file_id, payload in payloads.items()
        if payload.wav_bytes is None
    }
    items: dict[uuid.UUID, AudioFile] = {}
    resolved_store = None
    if binary_payloads:
        resolved_store = _object_store(session, store)
        items.update(
            bulk_update_packed_audio_files(
                session,
                resolved_store,
                binary_payloads,
                config,
                commit=False,
            )
        )
        waveform_crud.bulk_delete_waveforms(session, list(binary_payloads), commit=False)
    if metadata_payloads:
        metadata_items = get_audio_files_bulk(session, list(metadata_payloads))
        for audio_file_id, payload in metadata_payloads.items():
            item = metadata_items[audio_file_id]
            _update_audio_metadata(item, payload)
            items[audio_file_id] = item
    if items:
        session.commit()
    if resolved_store is not None:
        prune_fragmented_audio_packs(session, resolved_store, config)
    return items


def prune_audio_packs(
    session: Session,
    store: ObjectStore | None = None,
    config: AudioPackConfig = AudioPackConfig(),
) -> None:
    prune_fragmented_audio_packs(session, _object_store(session, store), config)


def _object_store(session: Session, store: ObjectStore | None) -> ObjectStore:
    if store is not None:
        return store
    return S3ObjectStore(settings_crud.object_store_config(session))


def _update_audio_metadata(item: AudioFile, payload: AudioUpdate) -> None:
    item.name = payload.name
    item.duration = payload.duration
    if "score" in payload.model_fields_set:
        item.score = payload.score
    if "language" in payload.model_fields_set:
        item.language = payload.language
    if "style_prompt" in payload.model_fields_set:
        item.style_prompt = payload.style_prompt
    if "voice_prompt" in payload.model_fields_set:
        item.voice_prompt = payload.voice_prompt
    item.segments = payload.segments
    item.metadata_ = payload.metadata
    item.virtual = payload.virtual
    item.updated_at = _now()


def _now() -> datetime:
    return datetime.now(UTC)
