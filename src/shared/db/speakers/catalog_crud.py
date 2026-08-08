from collections.abc import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile, AudioSegment
from shared.db.datasets.models import dataset_audio_files
from shared.db.speakers.schemas import SpeakerRead


def search_speakers(
    session: Session,
    query: str,
    limit: int,
    offset: int,
) -> tuple[list[SpeakerRead], int]:
    speaker_filter = AudioFile.speaker_id.is_not(None)
    if query:
        speaker_filter = speaker_filter & AudioFile.speaker_id.ilike(f"%{query}%")
    statement = (
        select(
            AudioFile.speaker_id,
            func.count(AudioFile.id).label("audio_files"),
            func.sum(AudioFile.segment_count).label("segments"),
        )
        .where(speaker_filter)
        .group_by(AudioFile.speaker_id)
        .order_by(AudioFile.speaker_id)
        .limit(limit)
        .offset(offset)
    )
    rows = session.execute(statement).all()
    speaker_ids = [row.speaker_id for row in rows]
    datasets = _speaker_datasets(session, speaker_ids)
    total = session.scalar(
        select(func.count(func.distinct(AudioFile.speaker_id))).where(speaker_filter)
    )
    return [
        SpeakerRead(
            id=row.speaker_id,
            audio_files=row.audio_files,
            segments=int(row.segments or 0),
            datasets=datasets[row.speaker_id],
        )
        for row in rows
    ], int(total or 0)


def rename_speaker(session: Session, speaker_id: str, replacement: str) -> None:
    if not replacement:
        raise ValueError("replacement speaker_id must not be empty")
    _replace_speaker(session, speaker_id, replacement)


def clear_speaker(session: Session, speaker_id: str) -> None:
    _replace_speaker(session, speaker_id, None)


def clear_matching_speakers(session: Session, query: str) -> None:
    rows, _total = search_speakers(session, query, 200, 0)
    while rows:
        for row in rows:
            _replace_speaker(session, row.id, None, commit=False)
        session.flush()
        rows, _total = search_speakers(session, query, 200, 0)
    session.commit()


def _speaker_datasets(session: Session, speaker_ids: Sequence[str]) -> dict[str, list]:
    datasets = {speaker_id: [] for speaker_id in speaker_ids}
    if not speaker_ids:
        return datasets
    statement = (
        select(AudioFile.speaker_id, dataset_audio_files.c.dataset_id)
        .join(dataset_audio_files, dataset_audio_files.c.audio_file_id == AudioFile.id)
        .where(AudioFile.speaker_id.in_(speaker_ids))
        .distinct()
    )
    for speaker_id, dataset_id in session.execute(statement):
        datasets[speaker_id].append(dataset_id)
    return datasets


def _replace_speaker(
    session: Session,
    speaker_id: str,
    replacement: str | None,
    commit: bool = True,
) -> None:
    audio_ids = set(session.scalars(
        select(AudioFile.id).where(AudioFile.speaker_id == speaker_id)
    ))
    segment_audio_ids = set(session.scalars(
        select(AudioSegment.audio_file_id).where(AudioSegment.speaker_id == speaker_id)
    ))
    if not audio_ids and not segment_audio_ids:
        raise KeyError(f"speaker not found: {speaker_id}")
    session.execute(
        update(AudioFile)
        .where(AudioFile.id.in_(audio_ids))
        .values(speaker_id=replacement)
    )
    session.execute(
        update(AudioSegment)
        .where(AudioSegment.speaker_id == speaker_id)
        .values(speaker_id=replacement)
    )
    if commit:
        session.commit()
