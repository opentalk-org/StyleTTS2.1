from __future__ import annotations

import io
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AUDIO, SAVE_RESULT
from runner.nodes.models import Audio, AudioRecordRef, AudioSegment, SaveResult, SegmentGroup, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import AudioCreate
from shared.db.datasets import crud as dataset_crud

Mode = Literal["create_new", "replace_all"]


class PersistSplitAudioRecordsSettings(StrictSettings):
    target_dataset_id: UUID | None = None
    source_dataset_id: UUID | None = None
    mode: Mode = "create_new"
    delete_source_on_replace: bool = False
    virtual: bool = False


class ExtractSegmentGroupAudioNode(Node):
    NODE_TYPE = "ExtractSegmentGroupAudio"
    CATEGORY = "Audio"
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                group = _group_from_audio(audio)
                source_id = _group_source_audio_id(group)
                item = audio_crud.get_audio_file(session, source_id)
                data = audio_crud.read_audio_file(session, source_id)
                source = _source_audio(item, data, group)
                outputs.append({"audio": extract_group_audio(source, group)})
        return outputs


class PersistSplitAudioRecordsNode(Node):
    NODE_TYPE = "PersistSplitAudioRecords"
    CATEGORY = "Audio"
    SETTINGS = PersistSplitAudioRecordsSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {
        "audio": Port("audio", AUDIO),
        "save_result": Port("save_result", SAVE_RESULT),
    }
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                assert audio.data is not None, f"audio bytes are required: {audio.id}"
                source_group_id = str(audio.metadata["source_group_id"])
                payload = AudioCreate(
                    name=audio.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    score=_target_audio_score(audio),
                    segments=[],
                    metadata=_target_audio_metadata(audio),
                    virtual=self.settings.virtual,
                )
                item = audio_crud.create_audio_file(session, payload)
                segments = audio_crud.replace_audio_segments(session, item.id, _metadata_segment_payloads(audio))
                _attach_datasets(session, item.id, self.settings)
                _replace_source_records(session, audio, self.settings)
                saved_group = _saved_segment_group(item, audio, segments)
                saved_audio = replace(
                    audio,
                    audio_file_id=item.id,
                    name=item.name,
                    id=stable_id("audio", item.id, item.name),
                    lineage_id=stable_id("audio_ref", item.id),
                    segments=saved_group.segments,
                    metadata=item.metadata_,
                    byte_length=item.byte_length,
                    virtual=item.virtual,
                )
                outputs.append({
                    "audio": saved_audio,
                    "save_result": _save_result(item, audio.lineage_id, source_group_id, len(segments)),
                })
        return outputs


def _group_from_audio(audio: Audio) -> SegmentGroup:
    return SegmentGroup(audio.name, audio.segments, stable_id("segment_group", audio.id, *(segment.id for segment in audio.segments)), audio.lineage_id, audio.metadata)


def extract_group_audio(audio: Audio, group: SegmentGroup) -> Audio:
    info = _read_wav_info(audio.data)
    span_start, span_end = group_span_bounds(group)
    start_frame = _seconds_to_frame(span_start, info["sample_rate"])
    end_frame = _seconds_to_frame(span_end, info["sample_rate"])
    assert 0 <= start_frame < end_frame <= info["frame_count"], f"group span outside audio bounds: {group.id}"
    data = _extract_wav_frames(audio.data, start_frame, end_frame)
    duration = (end_frame - start_frame) / float(info["sample_rate"])
    audio_id = stable_id("audio", audio.audio_file_id, group.id, span_start, span_end)
    metadata = {
        **audio.metadata,
        **group.metadata,
        "source_audio_id": str(audio.audio_file_id),
        "source_audio_lineage_id": audio.lineage_id,
        "source_group_id": group.id,
        "source_group_lineage_id": group.lineage_id,
        "span_start": span_start,
        "span_end": span_end,
        "split_segment_payloads": adjusted_segment_payloads(group),
    }
    return Audio(
        audio.audio_file_id,
        group.name,
        data,
        int(info["sample_rate"]),
        int(info["channels"]),
        0.0,
        duration,
        audio.confidence,
        audio_id,
        group.lineage_id,
        metadata,
    )


def adjusted_segment_payloads(group: SegmentGroup) -> list[dict[str, Any]]:
    span_start, _span_end = group_span_bounds(group)
    payloads = []
    for index, segment in enumerate(group.segments):
        start = max(0.0, segment.start - span_start)
        end = max(start, segment.end - span_start)
        payloads.append({
            "id": stable_id("segment", group.id, index, segment.id, segment.segment_id or ""),
            "start": start,
            "end": end,
            "text": segment.text,
            "phon": segment.phon,
            "speaker": segment.speaker or "",
            "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
            "type_": str(segment.metadata.get("type_", segment.metadata.get("model", "manual"))),
            "metadata": {
                **segment.metadata,
                "type_": str(segment.metadata.get("type_", segment.metadata.get("model", "manual"))),
                "source_audio_id": str(segment.source_audio_id),
                "source_segment_id": segment.segment_id or segment.id,
                "source_segment_lineage_id": segment.lineage_id,
            },
        })
    return payloads


def group_span_bounds(group: SegmentGroup) -> tuple[float, float]:
    assert group.segments, f"segment group is empty: {group.id}"
    return min(segment.start for segment in group.segments), max(segment.end for segment in group.segments)


def _group_source_audio_id(group: SegmentGroup) -> UUID:
    source_ids = {segment.source_audio_id for segment in group.segments}
    assert len(source_ids) == 1, f"group has multiple source audio ids: {group.id}"
    return next(iter(source_ids))


def _source_audio(item: AudioFile, data: bytes, group: SegmentGroup) -> Audio:
    info = _read_wav_info(data)
    metadata = {
        **item.metadata_,
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
    }
    audio_id = stable_id("audio", item.id, item.name)
    return Audio(
        item.id,
        item.name,
        data,
        int(metadata["sample_rate"]),
        int(metadata["channels"]),
        0.0,
        item.duration,
        1.0,
        audio_id,
        group.lineage_id,
        metadata,
    )


def _read_wav_info(data: bytes) -> dict[str, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            return {
                "sample_rate": source.getframerate(),
                "channels": source.getnchannels(),
                "sample_width": source.getsampwidth(),
                "frame_count": source.getnframes(),
            }
    except wave.Error as exc:
        raise ValueError("ExtractSegmentGroupAudio supports WAV audio bytes only") from exc


def _extract_wav_frames(data: bytes, start_frame: int, end_frame: int) -> bytes:
    source_buffer = io.BytesIO(data)
    output_buffer = io.BytesIO()
    with wave.open(source_buffer, "rb") as source:
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        with wave.open(output_buffer, "wb") as target:
            target.setnchannels(source.getnchannels())
            target.setsampwidth(source.getsampwidth())
            target.setframerate(source.getframerate())
            target.writeframes(frames)
    return output_buffer.getvalue()


def _seconds_to_frame(seconds: float, sample_rate: int) -> int:
    return int(round(max(0.0, seconds) * sample_rate))


def _target_audio_metadata(audio: Audio) -> dict[str, Any]:
    return {
        **audio.metadata,
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
    }


def _target_audio_score(audio: Audio) -> float | None:
    for key in ("score", "mos_score"):
        value = audio.metadata.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return None


def _attach_datasets(session, audio_file_id: UUID, settings: PersistSplitAudioRecordsSettings) -> None:
    dataset_ids = []
    if settings.target_dataset_id is not None:
        dataset_ids.append(settings.target_dataset_id)
    if settings.source_dataset_id is not None:
        dataset_ids.append(settings.source_dataset_id)
    for dataset_id in dict.fromkeys(dataset_ids):
        dataset_crud.add_audio_file_to_dataset(session, dataset_id, audio_file_id)


def _replace_source_records(session, audio: Audio, settings: PersistSplitAudioRecordsSettings) -> None:
    if settings.mode != "replace_all":
        return
    group_index = int(audio.metadata["group_index"])
    group_count = int(audio.metadata["group_count"])
    if group_index + 1 < group_count:
        return
    source_audio_id = UUID(str(audio.metadata["source_audio_id"]))
    if settings.source_dataset_id is not None:
        dataset_crud.remove_audio_file_from_dataset(session, settings.source_dataset_id, source_audio_id)
    if settings.delete_source_on_replace:
        audio_crud.delete_audio_file(session, source_audio_id)


def _saved_segment_group(item: AudioFile, audio: Audio, segments: list[dict[str, Any]]) -> SegmentGroup:
    ref = _audio_ref(item)
    saved_segments = [_segment_from_payload(ref, payload) for payload in segments]
    source_group_id = str(audio.metadata["source_group_id"])
    group_id = stable_id("segment_group", item.id, source_group_id)
    metadata = {
        "source_group_id": source_group_id,
        "source_group_lineage_id": audio.metadata["source_group_lineage_id"],
    }
    return SegmentGroup(audio.name, saved_segments, group_id, audio.lineage_id, metadata)


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
        metadata=metadata,
    )


def _audio_ref(item: AudioFile) -> AudioRecordRef:
    return AudioRecordRef(item.id, item.name, item.duration, item.byte_length, item.virtual, item.metadata_)


def _metadata_segment_payloads(audio: Audio) -> list[dict[str, Any]]:
    payloads = audio.metadata["split_segment_payloads"]
    assert isinstance(payloads, list), f"split segment payloads are required: {audio.id}"
    return payloads




def _save_result(item: AudioFile, lineage_id: str, source_group_id: str, segment_count: int) -> SaveResult:
    path = Path(f"db/audio/{item.id}")
    metadata = {"source_group_id": source_group_id, "segment_count": segment_count}
    return SaveResult(path, "split_audio_record", stable_id("save", path, source_group_id), lineage_id, metadata)
