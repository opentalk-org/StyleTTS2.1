import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Text, cast, desc, func, literal_column, or_, select, text
from sqlalchemy.orm import Session, defer, lazyload, selectinload

from shared.audio_annotations import AudioAnnotations
from shared.db.audio.models import AudioFile
from shared.db.audio.catalog_pagination import AudioCursor, cursor_filter, cursor_for_row
from shared.db.audio.schemas import (
    AudioBucketLocation,
    AudioFileReference,
)
from shared.db.common import many, one
from shared.db.datasets.models import Dataset, dataset_audio_files


AUDIO_LOCATION_QUERY_BATCH_SIZE = 50_000


def audio_file_annotations(item: AudioFile) -> AudioAnnotations:
    return AudioAnnotations(
        speaker_id=item.speaker_id,
        score=item.score,
        accuracy=item.accuracy,
        metadata=dict(item.metadata_),
    )


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
    cursor: str | None,
    preview_limit: int = 8,
    language: str = "",
) -> tuple[list[tuple[AudioFile, int, list[dict[str, Any]], int | None]], str | None, bool]:
    """List audio files without loading their potentially large segments column."""
    filters = _audio_filters(query, dataset, language)
    order = _audio_sort(sort)
    segment_count = func.coalesce(
        func.jsonb_array_length(AudioFile.segments),
        0,
    ).label("segment_count")
    preview_path = text(f"'$[0 to {max(preview_limit - 1, 0)}]'::jsonpath")
    segment_preview = literal_column(
        "(SELECT coalesce(jsonb_agg((segment - 'alignment' - 'annotations' - 'text' - 'phon') || "
        "jsonb_build_object('alignment', 'null'::jsonb, "
        "'annotations', (segment -> 'annotations') - 'metadata', "
        "'text', left(segment ->> 'text', 500), 'phon', left(segment ->> 'phon', 500))), '[]'::jsonb) "
        f"FROM jsonb_array_elements(jsonb_path_query_array(audio_files.segments, {preview_path.text})) AS preview(segment))"
    ).label("segment_preview")
    sample_rate = AudioFile.metadata_["sample_rate"].astext.label("sample_rate")
    page_cursor = AudioCursor.decode(cursor, sort) if cursor is not None else None

    if sort == "segments":
        statement = select(
            AudioFile,
            segment_count,
            segment_preview,
            sample_rate,
        ).options(
            defer(AudioFile.segments),
            defer(AudioFile.metadata_),
            lazyload(AudioFile.bucket_file),
            selectinload(AudioFile.datasets),
        )
        for item in filters:
            statement = statement.where(item)
        if page_cursor is not None:
            statement = statement.where(cursor_filter(page_cursor))
        statement = statement.order_by(*order).limit(limit + 1)
    else:
        page = select(AudioFile.id)
        for item in filters:
            page = page.where(item)
        if page_cursor is not None:
            page = page.where(cursor_filter(page_cursor))
        page = page.order_by(*order).limit(limit + 1).subquery()
        statement = (
            select(AudioFile, segment_count, segment_preview, sample_rate)
            .join(page, AudioFile.id == page.c.id)
            .options(
                defer(AudioFile.segments),
                defer(AudioFile.metadata_),
                lazyload(AudioFile.bucket_file),
                selectinload(AudioFile.datasets),
            )
            .order_by(*order)
        )

    rows = [
        (row[0], row[1], list(row[2] or []), int(row[3]) if row[3] is not None else None)
        for row in session.execute(statement).all()
    ]
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = cursor_for_row(sort, rows[-1][0], rows[-1][1]).encode() if has_more else None
    return rows, next_cursor, has_more


def search_audio_file_ids(
    session: Session,
    query: str,
    dataset: str,
    language: str = "",
) -> list[uuid.UUID]:
    statement = select(AudioFile.id)
    for item in _audio_filters(query, dataset, language):
        statement = statement.where(item)
    return list(session.execute(statement).scalars().all())


def get_audio_file(session: Session, audio_file_id: uuid.UUID) -> AudioFile:
    return get_audio_files_bulk(session, [audio_file_id])[audio_file_id]


def get_audio_files_bulk(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, AudioFile]:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return {}
    loaded = {}
    for start in range(0, len(ids), AUDIO_LOCATION_QUERY_BATCH_SIZE):
        batch = ids[start : start + AUDIO_LOCATION_QUERY_BATCH_SIZE]
        statement = select(AudioFile).where(AudioFile.id.in_(batch))
        loaded.update(
            {
                item.id: item
                for item in session.execute(statement).unique().scalars().all()
            }
        )
    missing_ids = set(ids).difference(loaded)
    if missing_ids:
        missing = sorted(str(audio_file_id) for audio_file_id in missing_ids)
        raise KeyError(f"Audio files not found: {missing}")
    return {audio_file_id: loaded[audio_file_id] for audio_file_id in ids}


def audio_bucket_locations(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
) -> list[AudioBucketLocation]:
    rows = {}
    for start in range(0, len(audio_file_ids), AUDIO_LOCATION_QUERY_BATCH_SIZE):
        batch = audio_file_ids[start : start + AUDIO_LOCATION_QUERY_BATCH_SIZE]
        statement = select(
            AudioFile.id,
            AudioFile.bucket_file_id,
            AudioFile.byte_length,
        ).where(AudioFile.id.in_(batch))
        rows.update(
            {
                audio_id: (bucket_file_id, byte_length)
                for audio_id, bucket_file_id, byte_length in session.execute(statement)
            }
        )
    return [
        AudioBucketLocation(
            audio_file_id=audio_file_id,
            bucket_file_id=rows[audio_file_id][0],
            byte_length=rows[audio_file_id][1],
        )
        for audio_file_id in audio_file_ids
    ]


def count_audio_file_references(
    session: Session,
    dataset_id: uuid.UUID | None,
    audio_file_ids: Sequence[uuid.UUID] | None,
    include_virtual: bool,
) -> int:
    if dataset_id is not None:
        one(session, Dataset, dataset_id)
    statement = select(func.count()).select_from(AudioFile)
    statement = _scoped_statement(
        statement,
        dataset_id,
        audio_file_ids,
        include_virtual,
    )
    return int(session.execute(statement).scalar_one())


def list_audio_file_references_page(
    session: Session,
    dataset_id: uuid.UUID | None,
    audio_file_ids: Sequence[uuid.UUID] | None,
    include_virtual: bool,
    after_id: uuid.UUID | None,
    limit: int,
) -> list[AudioFileReference]:
    statement = select(
        AudioFile.id,
        AudioFile.name,
        AudioFile.duration,
        AudioFile.speaker_id,
        AudioFile.score,
        AudioFile.accuracy,
        AudioFile.language,
        AudioFile.metadata_,
        AudioFile.byte_length,
        AudioFile.virtual,
        AudioFile.style_prompt,
        AudioFile.voice_prompt,
    )
    statement = _scoped_statement(
        statement,
        dataset_id,
        audio_file_ids,
        include_virtual,
    )
    if after_id is not None:
        statement = statement.where(AudioFile.id > after_id)
    rows = session.execute(statement.order_by(AudioFile.id).limit(limit)).all()
    return [
        AudioFileReference(
            id=row.id,
            name=row.name,
            duration=row.duration,
            language=row.language,
            annotations=AudioAnnotations(
                speaker_id=row.speaker_id,
                score=row.score,
                accuracy=row.accuracy,
                metadata=dict(row.metadata_),
            ),
            byte_length=row.byte_length,
            virtual=row.virtual,
            style_prompt=row.style_prompt,
            voice_prompt=row.voice_prompt,
        )
        for row in rows
    ]


def _audio_filters(query: str, dataset: str, language: str = "") -> list[Any]:
    filters = []
    if query:
        pattern = f"%{query}%"
        filters.append(
            or_(
                AudioFile.name.ilike(pattern),
                cast(AudioFile.metadata_, Text).ilike(pattern),
            )
        )
    if language.strip():
        filters.append(
            func.lower(AudioFile.language) == language.strip().lower()
        )
    if dataset == "unassigned":
        filters.append(~AudioFile.datasets.any())
    elif dataset != "all":
        dataset_id = uuid.UUID(dataset)
        filters.append(AudioFile.datasets.any(Dataset.id == dataset_id))
    return filters


def _audio_sort(sort: str):
    if sort == "name":
        return AudioFile.name, AudioFile.id
    if sort == "duration":
        return desc(AudioFile.duration), AudioFile.id
    if sort == "segments":
        return desc(func.jsonb_array_length(AudioFile.segments)), AudioFile.id
    if sort == "speaker_id":
        return AudioFile.speaker_id, AudioFile.id
    return desc(AudioFile.updated_at), AudioFile.id


def _scoped_statement(
    statement,
    dataset_id: uuid.UUID | None,
    audio_file_ids: Sequence[uuid.UUID] | None,
    include_virtual: bool,
):
    if dataset_id is not None:
        statement = statement.join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        ).where(dataset_audio_files.c.dataset_id == dataset_id)
    if audio_file_ids is not None:
        statement = statement.where(AudioFile.id.in_(audio_file_ids))
    if not include_virtual:
        statement = statement.where(AudioFile.virtual.is_(False))
    return statement
