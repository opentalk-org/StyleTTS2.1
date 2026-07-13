from pathlib import Path
from typing import Any
from uuid import UUID

from runner.nodes.models import Audio, AudioRecordRef, AudioSegment, SaveResult, SegmentGroup, stable_id
from shared.db.audio.models import AudioFile


def metadata_segment_payloads(audio: Audio) -> list[dict[str, Any]]:
    payloads = audio.metadata["split_segment_payloads"]
    assert isinstance(payloads, list), f"split segment payloads are required: {audio.id}"
    return payloads


def saved_segment_group(
    item: AudioFile,
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
    return SegmentGroup(audio.name, saved_segments, group_id, audio.lineage_id, metadata)


def save_result(
    item: AudioFile,
    lineage_id: str,
    source_group_id: str,
    segment_count: int,
) -> SaveResult:
    path = Path(f"db/audio/{item.id}")
    metadata = {"source_group_id": source_group_id, "segment_count": segment_count}
    return SaveResult(path, "split_audio_record", stable_id("save", path, source_group_id), lineage_id, metadata)


def _segment_from_payload(ref: AudioRecordRef, payload: dict[str, Any]) -> AudioSegment:
    segment_id = str(payload["id"])
    metadata = dict(payload["metadata"]) if "metadata" in payload else {}
    return AudioSegment(
        source_audio_id=ref.audio_file_id,
        name=ref.name,
        start=float(payload["start"]),
        end=float(payload["end"]),
        sample_rate=int(ref.metadata["sample_rate"]),
        channels=int(ref.metadata["channels"]),
        text=str(payload["text"]),
        phon=str(payload["phon"]),
        id=stable_id("segment", ref.audio_file_id, segment_id),
        lineage_id=stable_id("segment_lineage", ref.audio_file_id, segment_id),
        segment_id=segment_id,
        speaker=str(payload["speaker"]) if "speaker" in payload else None,
        voice_id=UUID(str(payload["voice_id"])) if payload.get("voice_id") else None,
        confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
        metadata=metadata,
    )


def _audio_ref(item: AudioFile) -> AudioRecordRef:
    return AudioRecordRef(item.id, item.name, item.duration, item.byte_length, item.virtual, item.metadata_)
