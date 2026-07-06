from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AUDIO, AUDIO_REF, AUDIO_SEGMENT, SAVE_RESULT, SEGMENT_GROUP, TEXT
from runner.nodes.models import Audio, AudioRecordRef, AudioSegment, SaveResult, SegmentGroup, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import AudioCreate, AudioUpdate


class SaveAudioRecordSettings(StrictSettings):
    virtual: bool = False


class SaveAudioRecordNode(Node):
    NODE_TYPE = "SaveAudioRecord"
    CATEGORY = "Audio / Writeback"
    SETTINGS = SaveAudioRecordSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=8, max_size=32)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                audio: Audio = inputs["audio"]
                payload = AudioCreate(
                    name=audio.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    segments=[],
                    metadata=_audio_metadata(audio),
                    virtual=self.settings.virtual,
                )
                item = audio_crud.create_audio_file(session, payload)
                outputs.append(_audio_writeback_output(item, audio.lineage_id, "created"))
        return outputs


class UpdateAudioRecordBytesNode(Node):
    NODE_TYPE = "UpdateAudioRecordBytes"
    CATEGORY = "Audio / Writeback"
    INPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF), "audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=8, max_size=32)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                ref: AudioRecordRef = inputs["audio_ref"]
                audio: Audio = inputs["audio"]
                item = audio_crud.get_audio_file(session, ref.audio_file_id)
                payload = AudioUpdate(
                    name=item.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    segments=item.segments,
                    metadata=_audio_metadata(audio),
                    virtual=item.virtual,
                )
                updated = audio_crud.update_audio_file(session, ref.audio_file_id, payload)
                outputs.append(_audio_writeback_output(updated, audio.lineage_id, "updated"))
        return outputs


class LoadAudioSegmentsNode(Node):
    NODE_TYPE = "LoadAudioSegments"
    CATEGORY = "Audio / Segments"
    INPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF)}
    OUTPUTS = {"segment_group": Port("segment_group", SEGMENT_GROUP)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                ref: AudioRecordRef = inputs["audio_ref"]
                segments = audio_crud.list_audio_segments(session, ref.audio_file_id)
                audio_segments = [_audio_segment_from_dict(ref, segment) for segment in segments]
                group_id = stable_id("segment_group", ref.audio_file_id, *(segment.id for segment in audio_segments))
                outputs.append({
                    "segment_group": SegmentGroup(ref.name, audio_segments, group_id, group_id, ref.metadata),
                })
        return outputs


class SaveAudioSegmentsNode(Node):
    NODE_TYPE = "SaveAudioSegments"
    CATEGORY = "Audio / Segments"
    INPUTS = {"audio_ref": Port("audio_ref", AUDIO_REF), "segment_group": Port("segment_group", SEGMENT_GROUP)}
    OUTPUTS = {"segment_group": Port("segment_group", SEGMENT_GROUP), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=8, max_size=32)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                ref: AudioRecordRef = inputs["audio_ref"]
                group: SegmentGroup = inputs["segment_group"]
                _assert_group_target(ref, group)
                segments = audio_crud.replace_audio_segments(
                    session,
                    ref.audio_file_id,
                    [_segment_dict(segment) for segment in group.segments],
                )
                saved = [_audio_segment_from_dict(ref, segment) for segment in segments]
                saved_group = SegmentGroup(group.name, saved, group.id, group.lineage_id, group.metadata)
                outputs.append({
                    "segment_group": saved_group,
                    "save_result": _save_result(f"db/audio/{ref.audio_file_id}/segments", "audio_segments", group.lineage_id, {"count": len(saved)}),
                })
        return outputs


class UpdateSegmentTextNode(Node):
    NODE_TYPE = "UpdateSegmentText"
    CATEGORY = "Audio / Segments"
    INPUTS = {"segment": Port("segment", AUDIO_SEGMENT), "text": Port("text", TEXT)}
    OUTPUTS = {"segment": Port("segment", AUDIO_SEGMENT), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                segment: AudioSegment = inputs["segment"]
                text: str = inputs["text"]
                segment_id = _segment_entry_id(segment)
                audio_crud.update_segment_text(session, segment.source_audio_id, segment_id, text)
                updated = replace(segment, text=text, segment_id=segment_id)
                outputs.append(_segment_writeback_output(updated, "segment_text"))
        return outputs


class UpdateSegmentPhonemesNode(Node):
    NODE_TYPE = "UpdateSegmentPhonemes"
    CATEGORY = "Audio / Segments"
    INPUTS = {"segment": Port("segment", AUDIO_SEGMENT), "phon": Port("phon", TEXT)}
    OUTPUTS = {"segment": Port("segment", AUDIO_SEGMENT), "save_result": Port("save_result", SAVE_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                segment: AudioSegment = inputs["segment"]
                phon: str = inputs["phon"]
                segment_id = _segment_entry_id(segment)
                audio_crud.update_segment_phonemes(session, segment.source_audio_id, segment_id, phon)
                updated = replace(segment, phon=phon, segment_id=segment_id)
                outputs.append(_segment_writeback_output(updated, "segment_phonemes"))
        return outputs


def _audio_metadata(audio: Audio) -> dict[str, Any]:
    return {**audio.metadata, "sample_rate": audio.sample_rate, "channels": audio.channels}


def _audio_writeback_output(item: AudioFile, lineage_id: str, action: str) -> dict[str, AudioRecordRef | SaveResult]:
    ref = AudioRecordRef(item.id, item.name, item.duration, item.byte_length, item.virtual, item.metadata_)
    return {
        "audio_ref": ref,
        "save_result": _save_result(f"db/audio/{item.id}", "audio_record", lineage_id, {"action": action}),
    }


def _audio_segment_from_dict(ref: AudioRecordRef, segment: dict[str, Any]) -> AudioSegment:
    segment_id = str(segment["id"])
    speaker = str(segment["speaker"]) if "speaker" in segment else None
    metadata = dict(segment["metadata"]) if "metadata" in segment else {}
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
    return {
        "id": _segment_entry_id(segment),
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "speaker": segment.speaker or "",
        "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
        "metadata": segment.metadata,
    }


def _assert_group_target(ref: AudioRecordRef, group: SegmentGroup) -> None:
    for segment in group.segments:
        assert segment.source_audio_id == ref.audio_file_id, f"segment belongs to different audio record: {segment.id}"


def _segment_entry_id(segment: AudioSegment) -> str:
    return segment.segment_id or segment.id


def _optional_uuid(value: object) -> UUID | None:
    if value is None or value == "":
        return None
    return UUID(str(value))


def _segment_writeback_output(segment: AudioSegment, kind: str) -> dict[str, AudioSegment | SaveResult]:
    return {
        "segment": segment,
        "save_result": _save_result(f"db/audio/{segment.source_audio_id}/segments/{_segment_entry_id(segment)}", kind, segment.lineage_id, {}),
    }


def _save_result(path: str, kind: str, lineage_id: str, metadata: dict[str, Any]) -> SaveResult:
    result_id = stable_id("save", path, kind, lineage_id)
    return SaveResult(Path(path), kind, result_id, lineage_id, metadata)
