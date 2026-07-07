from __future__ import annotations

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
from shared.db.audio.schemas import AudioCreate, AudioUpdate


class SaveAudioRecordSettings(StrictSettings):
    virtual: bool = False


class SaveAudioSegmentsSettings(StrictSettings):
    mode: Literal["replace", "append"] = "replace"


class SaveAudioRecordNode(Node):
    NODE_TYPE = "SaveAudioRecord"
    CATEGORY = "Audio"
    SETTINGS = SaveAudioRecordSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=8, max_size=32)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                assert audio.data is not None, f"audio bytes are required: {audio.id}"
                payload = AudioCreate(
                    name=audio.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    segments=[],
                    metadata=_audio_metadata(audio),
                    virtual=self.settings.virtual,
                )
                item = audio_crud.create_audio_file(session, payload)
                outputs.append(_audio_writeback_output(item, audio, "created"))
        return outputs


class UpdateAudioRecordBytesNode(Node):
    NODE_TYPE = "UpdateAudioRecordBytes"
    CATEGORY = "Audio"
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=8, max_size=32)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                assert audio.data is not None, f"audio bytes are required: {audio.id}"
                item = audio_crud.get_audio_file(session, audio.audio_file_id)
                payload = AudioUpdate(
                    name=item.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    segments=item.segments,
                    metadata=_audio_metadata(audio),
                    virtual=item.virtual,
                )
                updated = audio_crud.update_audio_file(session, audio.audio_file_id, payload)
                outputs.append(_audio_writeback_output(updated, audio, "updated"))
        return outputs


class LoadAudioSegmentsNode(Node):
    NODE_TYPE = "LoadAudioSegments"
    CATEGORY = "Audio"
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                ref = _audio_ref_from_audio(audio)
                segments = audio_crud.list_audio_segments(session, ref.audio_file_id)
                audio_segments = [_audio_segment_from_dict(ref, segment) for segment in segments]
                outputs.append({"audio": replace(audio, segments=audio_segments)})
        return outputs


class SaveAudioSegmentsNode(Node):
    NODE_TYPE = "SaveAudioSegments"
    CATEGORY = "Audio"
    SETTINGS = SaveAudioSegmentsSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=8, max_size=32)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                ref = _audio_ref_from_audio(audio)
                group = _segment_group_from_audio(audio)
                segments = _save_group_segments(session, ref, group, self.settings.mode)
                saved = [_audio_segment_from_dict(ref, segment) for segment in segments]
                outputs.append({
                    "audio": replace(audio, segments=saved),
                    "save_result": _save_result(f"db/audio/{ref.audio_file_id}/segments", "audio_segments", group.lineage_id, {"count": len(saved)}),
                })
        return outputs


def _audio_metadata(audio: Audio) -> dict[str, Any]:
    return {**audio.metadata, "sample_rate": audio.sample_rate, "channels": audio.channels}


def _audio_writeback_output(item: AudioFile, source: Audio, action: str) -> dict[str, Audio | SaveResult]:
    ref = AudioRecordRef(item.id, item.name, item.duration, item.byte_length, item.virtual, item.metadata_)
    audio = replace(
        source,
        audio_file_id=item.id,
        name=item.name,
        end=item.duration,
        id=stable_id("audio", item.id, item.name),
        lineage_id=ref.lineage_id,
        metadata=item.metadata_,
        byte_length=item.byte_length,
        virtual=item.virtual,
    )
    return {
        "audio": audio,
        "save_result": _save_result(f"db/audio/{item.id}", "audio_record", source.lineage_id, {"action": action}),
    }


def _audio_ref_from_audio(audio: Audio) -> AudioRecordRef:
    return AudioRecordRef(audio.audio_file_id, audio.name, audio.duration, audio.byte_length, audio.virtual, audio.metadata)


def _segment_group_from_audio(audio: Audio) -> SegmentGroup:
    group_id = stable_id("segment_group", audio.id, *(segment.id for segment in audio.segments))
    return SegmentGroup(audio.name, audio.segments, group_id, audio.lineage_id, audio.metadata)


def _audio_segment_from_dict(ref: AudioRecordRef, segment: dict[str, Any]) -> AudioSegment:
    segment_id = str(segment["id"])
    speaker = str(segment["speaker"]) if "speaker" in segment else None
    metadata = dict(segment["metadata"]) if isinstance(segment.get("metadata"), dict) else {}
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
        speaker=speaker,
        voice_id=_optional_uuid(segment["voice_id"]) if "voice_id" in segment else None,
        metadata=metadata,
    )


def _segment_dict(segment: AudioSegment) -> dict[str, Any]:
    type_ = _segment_type({"metadata": segment.metadata})
    return {
        "id": _segment_entry_id(segment),
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "speaker": segment.speaker or "",
        "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
        "type_": type_,
        "metadata": {**segment.metadata, "type_": type_},
    }


def _save_group_segments(session: Any, ref: AudioRecordRef, group: SegmentGroup, mode: Literal["replace", "append"]) -> list[dict[str, Any]]:
    new_segments = [_segment_dict(segment) for segment in group.segments]
    if mode == "append":
        new_segments = [*audio_crud.list_audio_segments(session, ref.audio_file_id), *new_segments]
    return audio_crud.replace_audio_segments(session, ref.audio_file_id, sorted(new_segments, key=_segment_sort_key))


def _segment_sort_key(segment: dict[str, Any]) -> tuple[float, float, str, str]:
    return (
        float(segment["start"]),
        float(segment["end"]),
        _segment_type(segment),
        str(segment["id"]),
    )


def _segment_type(segment: dict[str, Any]) -> str:
    if segment.get("type_"):
        return str(segment["type_"])
    metadata = segment.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("type_"):
            return str(metadata["type_"])
        if metadata.get("model"):
            return str(metadata["model"])
    return "manual"


def _segment_entry_id(segment: AudioSegment) -> str:
    return segment.segment_id or segment.id


def _optional_uuid(value: object) -> UUID | None:
    if value is None or value == "":
        return None
    return UUID(str(value))


def _save_result(path: str, kind: str, lineage_id: str, metadata: dict[str, Any]) -> SaveResult:
    result_id = stable_id("save", path, kind, lineage_id)
    return SaveResult(Path(path), kind, result_id, lineage_id, metadata)
