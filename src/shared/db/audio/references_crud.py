import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import AudioFileReference
from shared.db.common import one
from shared.db.datasets.models import Dataset, dataset_audio_files
from shared.audio_annotations import AudioAnnotations


def count_audio_file_references(
    session: Session,
    dataset_id: uuid.UUID | None,
    audio_file_ids: Sequence[uuid.UUID] | None,
    include_virtual: bool,
) -> int:
    if dataset_id is not None:
        one(session, Dataset, dataset_id)
    statement = select(func.count()).select_from(AudioFile)
    statement = _scoped_statement(statement, dataset_id, audio_file_ids, include_virtual)
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
        AudioFile.metadata_,
        AudioFile.byte_length,
        AudioFile.virtual,
        AudioFile.style_prompt,
        AudioFile.voice_prompt,
    )
    statement = _scoped_statement(statement, dataset_id, audio_file_ids, include_virtual)
    if after_id is not None:
        statement = statement.where(AudioFile.id > after_id)
    rows = session.execute(statement.order_by(AudioFile.id).limit(limit)).all()
    return [
        AudioFileReference(
            id=row.id,
            name=row.name,
            duration=row.duration,
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


def _scoped_statement(statement, dataset_id, audio_file_ids, include_virtual):
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
