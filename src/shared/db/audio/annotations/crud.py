import uuid
from collections.abc import Iterator, Sequence

from sqlalchemy import String, Text, bindparam, cast, func, select, update
from sqlalchemy.dialects.postgresql import aggregate_order_by, array
from sqlalchemy.orm import Session

from shared.db.audio.annotations.schemas import AudioAnnotationRow, AudioAnnotationUpdate
from shared.db.audio.models import AudioFile
from shared.db.datasets.models import Dataset, dataset_audio_files


def audio_annotation_update_statement():
    return (
        update(AudioFile.__table__)
        .where(AudioFile.id == bindparam("audio_id"))
        .values(
            style_prompt=bindparam("new_style_prompt"),
            voice_prompt=bindparam("new_voice_prompt"),
            score=bindparam("new_score"),
            accuracy=bindparam("new_accuracy"),
        )
    )


def iter_audio_annotations(session: Session, batch_size: int = 2_000) -> Iterator[list[AudioAnnotationRow]]:
    if batch_size <= 0:
        raise ValueError("audio annotation batch size must be positive")
    dataset_names = (
        select(
            func.coalesce(
                func.array_agg(aggregate_order_by(Dataset.name, Dataset.name)),
                array([], type_=Text),
            )
        )
        .select_from(dataset_audio_files.join(Dataset, Dataset.id == dataset_audio_files.c.dataset_id))
        .where(dataset_audio_files.c.audio_file_id == AudioFile.id)
        .scalar_subquery()
    )
    last_id: uuid.UUID | None = None
    while True:
        statement = (
            select(
                AudioFile.id,
                dataset_names.label("datasets"),
                AudioFile.style_prompt,
                AudioFile.voice_prompt,
                AudioFile.score,
                AudioFile.accuracy,
                AudioFile.metadata_.label("metadata"),
                func.md5(cast(AudioFile.metadata_, String)).label("metadata_hash"),
                func.md5(cast(AudioFile.segments, String)).label("segments_hash"),
            )
            .order_by(AudioFile.id)
            .limit(batch_size)
        )
        if last_id is not None:
            statement = statement.where(AudioFile.id > last_id)
        rows = session.execute(statement).mappings().all()
        if not rows:
            return
        batch = [AudioAnnotationRow.model_validate(row) for row in rows]
        yield batch
        last_id = batch[-1].id


def bulk_update_audio_annotations(
    session: Session,
    updates: Sequence[AudioAnnotationUpdate],
    commit: bool = True,
) -> None:
    if not updates:
        raise ValueError("audio annotation update requires at least one item")
    payloads = [
        {
            "audio_id": item.id,
            "new_style_prompt": item.style_prompt,
            "new_voice_prompt": item.voice_prompt,
            "new_score": item.score,
            "new_accuracy": item.accuracy,
        }
        for item in updates
    ]
    session.execute(audio_annotation_update_statement(), payloads)
    if commit:
        session.commit()


def replace_audio_language(
    session: Session,
    source: str,
    target: str,
    commit: bool = True,
) -> int:
    if source == target:
        raise ValueError("source and target languages must differ")
    result = session.execute(
        update(AudioFile)
        .where(AudioFile.language == source)
        .values(language=target)
    )
    if commit:
        session.commit()
    return int(result.rowcount)
