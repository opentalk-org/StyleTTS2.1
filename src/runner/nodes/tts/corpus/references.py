from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
from uuid import UUID

import soundfile as sf

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.datasets import crud as dataset_crud
from shared.db.datasets.clickhouse.training import TtsReferenceCandidate


@dataclass(frozen=True, slots=True)
class RegisteredReference:
    stream_id: str
    language: str
    audio_file_id: UUID
    transcript: str
    sample_rate: int
    wav_bytes: bytes


def select_reference_rows(
    rows: Sequence[TtsReferenceCandidate],
    stream_languages: Mapping[str, str],
) -> dict[str, TtsReferenceCandidate]:
    selected: dict[str, TtsReferenceCandidate] = {}
    for row in rows:
        stream = str(row.audio.metadata["stream"])
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
    rows = dataset_crud.list_tts_reference_candidates(
        dataset_ids, tuple(stream_languages)
    )
    selected = select_reference_rows(rows, stream_languages)
    with database_session() as session:
        wav_by_id = audio_crud.bulk_read_audio_files(
            session,
            [row.audio.id for row in selected.values()],
        )
    return {
        stream: RegisteredReference(
            stream_id=stream,
            language=stream_languages[stream],
            audio_file_id=row.audio.id,
            transcript=str(row.segments[0]["text"]).strip(),
            sample_rate=int(sf.info(io.BytesIO(wav_by_id[row.audio.id])).samplerate),
            wav_bytes=wav_by_id[row.audio.id],
        )
        for stream, row in selected.items()
    }


def _eligible(row: TtsReferenceCandidate, language: str) -> bool:
    audio = row.audio
    if (
        audio.virtual
        or audio.storage_kind.value != "packed"
        or audio.byte_length <= 0
        or not 4.0 <= audio.duration <= 12.0
        or audio.language != language
        or str(audio.metadata["language"]) != language
        or len(row.segments) != 1
    ):
        return False
    segment = row.segments[0]
    return (
        float(segment["start"]) == 0.0
        and float(segment["end"]) == audio.duration
        and bool(str(segment["text"]).strip())
    )


def _rank(row: TtsReferenceCandidate) -> tuple[float, int, str]:
    return (
        abs(row.audio.duration - 8.0),
        int(row.audio.metadata["sentence_index"]),
        str(row.audio.id),
    )
