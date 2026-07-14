from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from shared.db.audio.rows_crud import get_audio_files_bulk
from shared.db.audio.segments_crud import bulk_replace_audio_segments


@dataclass(frozen=True)
class AcceptedSpeakerAssignment:
    audio_id: UUID
    segment_id: str
    voice_id: UUID


@dataclass(frozen=True)
class SpeakerAssignmentWriteCounts:
    accepted_assignment_count: int
    updated_audio_count: int


def bulk_apply_speaker_assignments(
    session: Session,
    assignments: Iterable[AcceptedSpeakerAssignment],
) -> SpeakerAssignmentWriteCounts:
    grouped: dict[UUID, dict[str, UUID]] = {}
    assignment_count = 0
    for assignment in assignments:
        audio_assignments = grouped.setdefault(assignment.audio_id, {})
        if assignment.segment_id in audio_assignments:
            raise ValueError(
                f"duplicate speaker assignment: {assignment.audio_id}/"
                f"{assignment.segment_id}"
            )
        audio_assignments[assignment.segment_id] = assignment.voice_id
        assignment_count += 1
    audio_rows = get_audio_files_bulk(session, list(grouped))
    replacements = {
        audio_id: _assigned_segments(audio_id, row.segments, grouped[audio_id])
        for audio_id, row in audio_rows.items()
    }
    bulk_replace_audio_segments(session, replacements)
    return SpeakerAssignmentWriteCounts(
        accepted_assignment_count=assignment_count,
        updated_audio_count=len(replacements),
    )


def _assigned_segments(
    audio_id: UUID,
    segments: list[dict[str, object]],
    assignments: dict[str, UUID],
) -> list[dict[str, object]]:
    missing = set(assignments)
    updated = []
    for segment in segments:
        segment_id = str(segment["id"])
        voice_id = assignments.get(segment_id)
        if voice_id is None:
            updated.append(segment)
            continue
        updated.append({**segment, "voice_id": str(voice_id)})
        missing.remove(segment_id)
    if missing:
        raise KeyError(
            f"audio file {audio_id} has missing segment IDs: {sorted(missing)}"
        )
    return updated
