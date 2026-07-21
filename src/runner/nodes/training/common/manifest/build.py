from __future__ import annotations

from dataclasses import dataclass
from itertools import batched
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from runner.nodes.audio_segments.writeback_helpers import audio_segment_from_dict
from runner.nodes.models import AudioRecordRef, AudioSegment
from shared.db import database_session
from shared.db.audio import crud as audio_crud


MATERIALIZE_BATCH_SIZE = 256


@dataclass(frozen=True)
class ManifestLine:
    audio_id: UUID
    value: str
    phon: str
    speaker_id: str


@dataclass(frozen=True)
class AudioFileSelection:
    dataset_id: UUID
    audio_file_ids: list[UUID]


def audio_file_selection(value: dict[str, Any]) -> AudioFileSelection:
    dataset_id = UUID(str(value["dataset_id"]))
    raw_ids = value["ids"]
    if not isinstance(raw_ids, list):
        raise TypeError("audio_file_ids.ids must be a list")
    return AudioFileSelection(dataset_id, [UUID(str(audio_file_id)) for audio_file_id in raw_ids])


def phoneme_alphabet_symbols(value: dict) -> list[str]:
    # Prefer an explicit symbol_list: the canonical StyleTTS2 alphabet contains a
    # literal space symbol, so a space-joined string cannot round-trip it.
    symbol_list = value.get("symbol_list") if isinstance(value, dict) else None
    if isinstance(symbol_list, list) and symbol_list:
        return [str(part) for part in symbol_list]
    symbols = value["symbols"] if "symbols" in value else ""
    if isinstance(symbols, str):
        return [part for part in symbols.split(" ") if part]
    if isinstance(symbols, list):
        return [str(part) for part in symbols]
    raise TypeError("phoneme alphabet symbols must be a string or list")


def segments_from_audio_file_ids(audio_file_ids: list[UUID]) -> list[AudioSegment]:
    segments: list[AudioSegment] = []
    with database_session() as session:
        items = audio_crud.get_audio_files_bulk(session, audio_file_ids)
        for audio_file_id, item in items.items():
            ref = AudioRecordRef(item.id, item.name, item.duration, item.byte_length, item.virtual, audio_crud.audio_file_annotations(item))
            segments.extend(
                _audio_segment_from_dict(ref, segment)
                for segment in item.segments
            )
    return segments


def usable_groups(segments: list[AudioSegment]) -> dict[UUID, list[AudioSegment]]:
    groups: dict[UUID, list[AudioSegment]] = {}
    for segment in sorted(segments, key=_segment_sort_key):
        if not segment.phon.strip() or segment.start >= segment.end:
            continue
        groups.setdefault(segment.source_audio_id, []).append(segment)
    return {audio_id: items for audio_id, items in groups.items() if items}


def manifest_lines(
    groups: dict[UUID, list[AudioSegment]],
    audio_ids: list[UUID],
    root_path: str,
    audio_dir: Path,
    speaker_ids: dict[str, int],
) -> list[ManifestLine]:
    lines: list[ManifestLine] = []
    for audio_id in audio_ids:
        segments = groups[audio_id]
        lines.append(
            ManifestLine(
                audio_id=audio_id,
                value=_manifest_audio_value(audio_id, root_path, audio_dir),
                phon=" ".join(segment.phon.strip() for segment in segments),
                speaker_id=str(speaker_ids[speaker_key(segments[0])]),
            )
        )
    return lines


def speaker_id_map(groups: dict[UUID, list[AudioSegment]]) -> dict[str, int]:
    """Map each distinct speaker key to a stable integer id.

    The StyleTTS2 dataset loader casts the manifest speaker column with
    ``int(...)`` and uses it as a multispeaker embedding index, so the column
    must be numeric. Free-text speaker IDs are enumerated to
    contiguous integers here."""
    keys = sorted({speaker_key(segments[0]) for segments in groups.values() if segments})
    return {key: index for index, key in enumerate(keys)}


def speaker_key(segment: AudioSegment) -> str:
    if segment.speaker_id is not None and segment.speaker_id.strip():
        return segment.speaker_id.strip()
    return "0"


def manifest_audio_dir(output_dir: Path, root_path: str) -> Path:
    if root_path:
        return Path(root_path)
    return output_dir / "audio"


def materialize_audio_files(audio_ids: list[UUID], audio_dir: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    with database_session() as session:
        for audio_id_batch in batched(audio_ids, MATERIALIZE_BATCH_SIZE):
            audio_data = audio_crud.bulk_read_audio_files(session, audio_id_batch)
            for audio_id, data in audio_data.items():
                target = audio_dir / f"{audio_id}.wav"
                target.write_bytes(data)


def write_manifest(path: Path, lines: list[ManifestLine]) -> None:
    content = "".join(f"{line.value}|{line.phon}|{line.speaker_id}\n" for line in lines)
    path.write_text(content, encoding="utf-8")


def write_text_sidecar(path: Path, groups: dict[UUID, list[AudioSegment]], audio_ids: list[UUID]) -> None:
    rows = [_segment_sidecar_row(segment) for audio_id in audio_ids for segment in groups[audio_id]]
    content = "".join(json.dumps(row, default=str, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _manifest_audio_value(audio_id: UUID, root_path: str, audio_dir: Path) -> str:
    filename = f"{audio_id}.wav"
    if root_path:
        return str(Path(root_path) / filename)
    return str((audio_dir / filename).resolve())


def _segment_sort_key(segment: AudioSegment) -> tuple[str, float, float, str]:
    segment_key = segment.segment_id if segment.segment_id is not None else segment.id
    return (str(segment.source_audio_id), segment.start, segment.end, segment_key)


def _audio_segment_from_dict(ref: AudioRecordRef, segment: dict[str, Any]) -> AudioSegment:
    return audio_segment_from_dict(ref, segment)


def _segment_sidecar_row(segment: AudioSegment) -> dict[str, object]:
    return {
        "audio_id": str(segment.source_audio_id),
        "segment_id": segment.segment_id,
        "runtime_segment_id": segment.id,
        "lineage_id": segment.lineage_id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "annotations": segment.annotations.model_dump(mode="json"),
    }
