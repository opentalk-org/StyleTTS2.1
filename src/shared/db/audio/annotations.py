import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.audio_annotations import AudioAnnotations
from shared.db.audio.catalog import get_audio_files_bulk
from shared.db.audio.models import AudioFile
from shared.db.audio.segments import bulk_replace_audio_segments


@dataclass(frozen=True)
class AcceptedSpeakerAssignment:
    audio_id: UUID
    segment_id: str
    speaker_id: str


@dataclass(frozen=True)
class SpeakerAssignmentWriteCounts:
    accepted_assignment_count: int
    updated_audio_count: int


def bulk_apply_speaker_assignments(
    session: Session,
    assignments: Iterable[AcceptedSpeakerAssignment],
) -> SpeakerAssignmentWriteCounts:
    grouped: dict[UUID, dict[str, str]] = {}
    assignment_count = 0
    for assignment in assignments:
        audio_assignments = grouped.setdefault(assignment.audio_id, {})
        if assignment.segment_id in audio_assignments:
            raise ValueError(
                f"duplicate speaker assignment: {assignment.audio_id}/"
                f"{assignment.segment_id}"
            )
        audio_assignments[assignment.segment_id] = assignment.speaker_id
        assignment_count += 1
    audio_rows = get_audio_files_bulk(session, list(grouped))
    replacements = {
        audio_id: _assigned_segments(
            audio_id,
            row.segments,
            grouped[audio_id],
        )
        for audio_id, row in audio_rows.items()
    }
    bulk_replace_audio_segments(session, replacements)
    return SpeakerAssignmentWriteCounts(
        accepted_assignment_count=assignment_count,
        updated_audio_count=len(replacements),
    )


def bulk_update_audio_scores(
    session: Session,
    scores: dict[uuid.UUID, float],
) -> dict[uuid.UUID, AudioFile]:
    if not scores:
        raise ValueError("audio score update requires at least one item")
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("audio scores must be finite")
    statement = select(AudioFile).where(AudioFile.id.in_(scores))
    items = {
        item.id: item
        for item in session.execute(statement).unique().scalars().all()
    }
    missing_ids = set(scores).difference(items)
    if missing_ids:
        missing = sorted(str(audio_id) for audio_id in missing_ids)
        raise KeyError(f"Audio files not found: {missing}")
    updated_at = datetime.now(UTC)
    for audio_id, score in scores.items():
        items[audio_id].score = score
        items[audio_id].updated_at = updated_at
    session.commit()
    return {audio_id: items[audio_id] for audio_id in scores}


def _assigned_segments(
    audio_id: UUID,
    segments: list[dict[str, object]],
    assignments: dict[str, str],
) -> list[dict[str, object]]:
    missing = set(assignments)
    updated = []
    for segment in segments:
        segment_id = str(segment["id"])
        speaker_id = assignments.get(segment_id)
        if speaker_id is None:
            updated.append(segment)
            continue
        annotations = AudioAnnotations.model_validate(segment["annotations"])
        updated.append(
            {
                **segment,
                "annotations": annotations.model_copy(
                    update={"speaker_id": speaker_id}
                ).model_dump(mode="json"),
            }
        )
        missing.remove(segment_id)
    if missing:
        raise KeyError(
            f"audio file {audio_id} has missing segment IDs: {sorted(missing)}"
        )
    return updated
