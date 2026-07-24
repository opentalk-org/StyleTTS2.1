from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.datasets import crud as dataset_crud


@dataclass(frozen=True, slots=True)
class RegisteredReference:
    stream_id: str
    language: str
    audio_file_id: UUID
    transcript: str
    wav_bytes: bytes


def select_reference_rows(
    rows: Sequence[AudioFile],
    stream_languages: Mapping[str, str],
) -> dict[str, AudioFile]:
    selected: dict[str, AudioFile] = {}
    for row in rows:
        stream = str(row.metadata_["stream"])
        language = stream_languages[stream]
        if not _eligible(row, language):
            continue
        current = selected.get(stream)
        if current is None or _rank(row) < _rank(current):
            selected[stream] = row
    missing = sorted(set(stream_languages) - set(selected))
    if missing:
        raise ValueError(f"tts_reference_missing:{','.join(missing)}")
    return selected


def load_registered_references(
    dataset_ids: Sequence[UUID],
    stream_languages: Mapping[str, str],
) -> dict[str, RegisteredReference]:
    with database_session() as session:
        rows = dataset_crud.list_tts_reference_candidates(
            session,
            dataset_ids,
            tuple(stream_languages),
        )
        selected = select_reference_rows(rows, stream_languages)
        wav_by_id = audio_crud.bulk_read_audio_files(
            session,
            [row.id for row in selected.values()],
        )
    return {
        stream: RegisteredReference(
            stream_id=stream,
            language=stream_languages[stream],
            audio_file_id=row.id,
            transcript=str(row.segments[0]["text"]).strip(),
            wav_bytes=wav_by_id[row.id],
        )
        for stream, row in selected.items()
    }


def _eligible(row: AudioFile, language: str) -> bool:
    if (
        row.virtual
        or row.storage_kind != "packed"
        or row.byte_length <= 0
        or not 4.0 <= row.duration <= 12.0
        or row.language != language
        or str(row.metadata_["language"]) != language
        or len(row.segments) != 1
    ):
        return False
    segment = row.segments[0]
    return (
        float(segment["start"]) == 0.0
        and float(segment["end"]) == row.duration
        and bool(str(segment["text"]).strip())
    )


def _rank(row: AudioFile) -> tuple[float, int, str]:
    return (
        abs(row.duration - 8.0),
        int(row.metadata_["sentence_index"]),
        str(row.id),
    )
