from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, SaveResultPort
from runner.nodes.models import Audio, AudioRecordRef, AudioSegment, SaveResult, SegmentGroup, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.models import AudioFile
from shared.db.audio.schemas import AudioCreate, AudioUpdate


class SaveAudioRecordSettings(StrictSettings):
    virtual: bool = False
    bulk_import_packs: bool = False


class SaveAudioSegmentsSettings(StrictSettings):
    mode: Literal["replace", "append"] = "replace"


class SaveAudioRecordNode(Node):
    NODE_TYPE = "SaveAudioRecord"
    CATEGORY = "Audio"
    SETTINGS = SaveAudioRecordSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "save_result": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        audios = []
        payloads = []
        for inputs in batch:
            context.check_cancel()
            audio: Audio = inputs["audio"]
            assert audio.data is not None, f"audio bytes are required: {audio.id}"
            audios.append(audio)
            payloads.append(
                AudioCreate(
                    name=audio.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    score=_audio_score(audio),
                    language=_audio_language(audio),
                    segments=[],
                    metadata=_audio_metadata(audio),
                    virtual=self.settings.virtual,
                )
            )
        with database_session() as session:
            items = audio_crud.bulk_create_audio_files(
                session,
                payloads,
                config=_audio_pack_config(self.settings),
            )
        context.check_cancel()
        return [_audio_writeback_output(item, audio, "created") for item, audio in zip(items, audios, strict=True)]


class UpdateAudioRecordBytesNode(Node):
    NODE_TYPE = "UpdateAudioRecordBytes"
    CATEGORY = "Audio"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "save_result": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                context.check_cancel()
                audio: Audio = inputs["audio"]
                assert audio.data is not None, f"audio bytes are required: {audio.id}"
                item = audio_crud.get_audio_file(session, audio.audio_file_id)
                payload = AudioUpdate(
                    name=item.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    score=_audio_score(audio, fallback=item.score),
                    language=_audio_language(audio, fallback=item.language),
                    segments=item.segments,
                    metadata=_audio_metadata(audio),
                    virtual=item.virtual,
                )
                updated = audio_crud.update_audio_file(session, audio.audio_file_id, payload)
                context.check_cancel()
                outputs.append(_audio_writeback_output(updated, audio, "updated"))
        return outputs


class LoadAudioSegmentsNode(Node):
    NODE_TYPE = "LoadAudioSegments"
    CATEGORY = "Audio"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        with database_session() as session:
            for inputs in batch:
                context.check_cancel()
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
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "save_result": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        records: list[tuple[Audio, AudioRecordRef, SegmentGroup, list[dict[str, Any]]]] = []
        with database_session() as session:
            for inputs in batch:
                context.check_cancel()
                audio: Audio = inputs["audio"]
                ref = _audio_ref_from_audio(audio)
                group = _segment_group_from_audio(audio)
                records.append((audio, ref, group, _new_group_segments(session, ref, group, self.settings.mode)))
            saved_by_id = audio_crud.bulk_replace_audio_segments(
                session,
                {ref.audio_file_id: segments for _, ref, _, segments in records},
            )
        outputs = []
        for audio, ref, group, _ in records:
            segments = saved_by_id[ref.audio_file_id]
            saved = [_audio_segment_from_dict(ref, segment) for segment in segments]
            outputs.append({
                "audio": replace(audio, segments=saved),
                "save_result": _save_result(f"db/audio/{ref.audio_file_id}/segments", "audio_segments", group.lineage_id, {"count": len(saved)}),
            })
        return outputs


def _audio_metadata(audio: Audio) -> dict[str, Any]:
    return {**audio.metadata, "sample_rate": audio.sample_rate, "channels": audio.channels}


def _audio_pack_config(settings: SaveAudioRecordSettings) -> audio_crud.AudioPackConfig:
    if not settings.bulk_import_packs:
        return audio_crud.AudioPackConfig()
    return audio_crud.AudioPackConfig(
        target_pack_bytes=512 * 1024 * 1024,
        reuse_open_packs=False,
        seal_on_flush=True,
    )


def _audio_score(audio: Audio, fallback: float | None = None) -> float | None:
    for key in ("score", "mos_score"):
        value = audio.metadata.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return fallback


def _audio_language(audio: Audio, fallback: str | None = None) -> str | None:
    value = audio.metadata.get("language")
    if value is None or value == "":
        return fallback
    return str(value)


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
        confidence=_optional_float(segment.get("confidence")),
        metadata=metadata,
        alignment=segment["alignment"] if isinstance(segment.get("alignment"), list) else None,
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
        "confidence": segment.confidence,
        "type_": type_,
        "metadata": {**segment.metadata, "type_": type_},
        "alignment": segment.alignment,
    }


def _new_group_segments(session: Any, ref: AudioRecordRef, group: SegmentGroup, mode: Literal["replace", "append"]) -> list[dict[str, Any]]:
    new_segments = [_segment_dict(segment) for segment in group.segments]
    if mode == "append":
        new_segments = [*audio_crud.list_audio_segments(session, ref.audio_file_id), *new_segments]
    return sorted(new_segments, key=_segment_sort_key)


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


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)  # type: ignore[arg-type]


def _save_result(path: str, kind: str, lineage_id: str, metadata: dict[str, Any]) -> SaveResult:
    result_id = stable_id("save", path, kind, lineage_id)
    return SaveResult(Path(path), kind, result_id, lineage_id, metadata)
