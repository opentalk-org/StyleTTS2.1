from pathlib import Path
from typing import Any

from runner.nodes.audio_segments.writeback_helpers import audio_segment_from_dict
from runner.nodes.models import (
    Audio,
    AudioRecordRef,
    AudioSegment,
    SaveResult,
    SegmentGroup,
    stable_id,
)
from shared.db.audio import crud as audio_crud
from shared.db.audio.clickhouse.models import AudioFileRecord


def metadata_segment_payloads(audio: Audio) -> list[dict[str, Any]]:
    payloads = audio.metadata["split_segment_payloads"]
    assert isinstance(payloads, list), (
        f"split segment payloads are required: {audio.id}"
    )
    return payloads


def saved_segment_group(
    item: AudioFileRecord,
    audio: Audio,
    segments: list[dict[str, Any]],
) -> SegmentGroup:
    ref = _audio_ref(item)
    saved_segments = [_segment_from_payload(ref, payload) for payload in segments]
    source_group_id = str(audio.metadata["source_group_id"])
    group_id = stable_id("segment_group", item.id, source_group_id)
    metadata = {
        "source_group_id": source_group_id,
        "source_group_lineage_id": audio.metadata["source_group_lineage_id"],
    }
    return SegmentGroup(
        audio.name, saved_segments, group_id, audio.lineage_id, metadata
    )


def save_result(
    item: AudioFileRecord,
    lineage_id: str,
    source_group_id: str,
    segment_count: int,
) -> SaveResult:
    path = Path(f"db/audio/{item.id}")
    metadata = {"source_group_id": source_group_id, "segment_count": segment_count}
    return SaveResult(
        path,
        "split_audio_record",
        stable_id("save", path, source_group_id),
        lineage_id,
        metadata,
    )


def _segment_from_payload(ref: AudioRecordRef, payload: dict[str, Any]) -> AudioSegment:
    return audio_segment_from_dict(ref, payload)


def _audio_ref(item: AudioFileRecord) -> AudioRecordRef:
    return AudioRecordRef(
        item.id,
        item.name,
        item.duration,
        item.byte_length,
        item.virtual,
        audio_crud.audio_file_annotations(item),
    )
