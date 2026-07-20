from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, true, tuple_
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.common import one
from shared.db.datasets.models import Dataset, dataset_audio_files
from shared.audio_annotations import AudioAnnotations


@dataclass(frozen=True)
class SegmentCursor:
    audio_file_id: UUID
    segment_index: int


@dataclass(frozen=True)
class SegmentReference:
    audio_file_id: UUID
    audio_name: str
    audio_duration: float
    annotations: AudioAnnotations
    audio_byte_length: int
    audio_virtual: bool
    audio_storage_kind: str
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    segment_index: int
    segment: dict[str, Any]

    @property
    def cursor(self) -> SegmentCursor:
        return SegmentCursor(self.audio_file_id, self.segment_index)


def count_segment_references(session: Session, dataset_id: UUID) -> int:
    one(session, Dataset, dataset_id)
    statement = (
        select(func.coalesce(func.sum(func.jsonb_array_length(AudioFile.segments)), 0))
        .select_from(AudioFile)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(dataset_audio_files.c.dataset_id == dataset_id)
    )
    return int(session.execute(statement).scalar_one())


def list_segment_references_page(
    session: Session,
    dataset_id: UUID,
    after: SegmentCursor | None,
    limit: int,
) -> list[SegmentReference]:
    rows = session.execute(segment_references_statement(dataset_id, after, limit)).all()
    return [
        SegmentReference(
            audio_file_id=row.audio_file_id,
            audio_name=row.audio_name,
            audio_duration=row.audio_duration,
            annotations=AudioAnnotations(
                speaker_id=row.speaker_id,
                voice_id=row.voice_id,
                score=row.score,
                accuracy=row.accuracy,
                metadata=dict(row.audio_metadata),
            ),
            audio_byte_length=row.audio_byte_length,
            audio_virtual=row.audio_virtual,
            audio_storage_kind=row.audio_storage_kind,
            language=row.language,
            style_prompt=row.style_prompt,
            voice_prompt=row.voice_prompt,
            segment_index=row.segment_index,
            segment=row.segment,
        )
        for row in rows
    ]


def segment_references_statement(
    dataset_id: UUID,
    after: SegmentCursor | None,
    limit: int,
):
    expanded = (
        func.jsonb_array_elements(AudioFile.segments)
        .table_valued("segment", with_ordinality="ordinality")
        .render_derived()
        .lateral()
    )
    segment_index = (expanded.c.ordinality - 1).label("segment_index")
    statement = (
        select(
            AudioFile.id.label("audio_file_id"),
            AudioFile.name.label("audio_name"),
            AudioFile.duration.label("audio_duration"),
            AudioFile.speaker_id,
            AudioFile.voice_id,
            AudioFile.score,
            AudioFile.accuracy,
            AudioFile.metadata_.label("audio_metadata"),
            AudioFile.byte_length.label("audio_byte_length"),
            AudioFile.virtual.label("audio_virtual"),
            AudioFile.storage_kind.label("audio_storage_kind"),
            AudioFile.language,
            AudioFile.style_prompt,
            AudioFile.voice_prompt,
            segment_index,
            expanded.c.segment,
        )
        .select_from(AudioFile)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .join(expanded, true())
        .where(dataset_audio_files.c.dataset_id == dataset_id)
    )
    if after is not None:
        statement = statement.where(
            tuple_(AudioFile.id, expanded.c.ordinality - 1)
            > tuple_(after.audio_file_id, after.segment_index)
        )
    return statement.order_by(AudioFile.id, segment_index).limit(limit)
