from pathlib import Path
from typing import Any, Literal

from runner.nodes.models import Audio, AudioRecordRef, AudioSegment, SaveResult, SegmentGroup, stable_id
from shared.audio_annotations import AudioAnnotations


def audio_ref_from_audio(audio: Audio) -> AudioRecordRef:
    return AudioRecordRef(audio.audio_file_id, audio.name, audio.duration, audio.byte_length, audio.virtual, audio.annotations)


def segment_group_from_audio(audio: Audio) -> SegmentGroup:
    group_id = stable_id("segment_group", audio.id, *(segment.id for segment in audio.segments))
    return SegmentGroup(audio.name, audio.segments, group_id, audio.lineage_id, audio.metadata)


def audio_segment_from_dict(ref: AudioRecordRef, segment: dict[str, Any]) -> AudioSegment:
    segment_id = str(segment["id"])
    annotations = AudioAnnotations.model_validate(segment["annotations"])
    metadata = dict(annotations.metadata)
    metadata.setdefault("type_", _segment_type(segment))
    return AudioSegment(
        source_audio_id=ref.audio_file_id,
        name=ref.name,
        start=float(segment["start"]),
        end=float(segment["end"]),
        sample_rate=int(ref.metadata["sample_rate"]),
        channels=int(ref.metadata["channels"]) if "channels" in ref.metadata else 1,
        text=str(segment["text"]),
        phon=str(segment["phon"]),
        id=stable_id("segment", ref.audio_file_id, segment_id),
        lineage_id=stable_id("segment_lineage", ref.audio_file_id, segment_id),
        segment_id=segment_id,
        annotations=annotations.model_copy(update={"metadata": metadata}),
        alignment=segment["alignment"] if isinstance(segment.get("alignment"), list) else None,
    )


def new_group_segments(
    group: SegmentGroup,
    mode: Literal["replace", "append"],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    new_segments = [_segment_dict(segment) for segment in group.segments]
    if mode == "append":
        new_segments = [*existing, *new_segments]
    return sorted(new_segments, key=_segment_sort_key)


def save_result(path: str, kind: str, lineage_id: str, metadata: dict[str, Any]) -> SaveResult:
    result_id = stable_id("save", path, kind, lineage_id)
    return SaveResult(Path(path), kind, result_id, lineage_id, metadata)


def _segment_dict(segment: AudioSegment) -> dict[str, Any]:
    type_ = _segment_type({"annotations": segment.annotations.model_dump(mode="json")})
    annotations = segment.annotations.model_copy(
        update={"metadata": {**segment.metadata, "type_": type_}}
    )
    return {
        "id": segment.segment_id or segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "annotations": annotations.model_dump(mode="json"),
        "type_": type_,
        "alignment": segment.alignment,
    }


def _segment_sort_key(segment: dict[str, Any]) -> tuple[float, float, str, str]:
    return float(segment["start"]), float(segment["end"]), _segment_type(segment), str(segment["id"])


def _segment_type(segment: dict[str, Any]) -> str:
    if segment.get("type_"):
        return str(segment["type_"])
    annotations = segment.get("annotations")
    metadata = annotations.get("metadata") if isinstance(annotations, dict) else None
    if isinstance(metadata, dict):
        if metadata.get("type_"):
            return str(metadata["type_"])
        if metadata.get("model"):
            return str(metadata["model"])
    return "manual"
