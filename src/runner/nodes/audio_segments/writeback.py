from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Literal
from uuid import UUID

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, SaveResultPort
from runner.nodes.models import Audio, AudioRecordRef, SaveResult, SegmentGroup, stable_id
from runner.nodes.audio_segments.writeback_helpers import (
    audio_ref_from_audio as _audio_ref_from_audio,
    audio_segment_from_dict as _audio_segment_from_dict,
    new_group_segments as _new_group_segments,
    save_result as _save_result,
    segment_group_from_audio as _segment_group_from_audio,
)
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
    DESCRIPTION = "Create a new audio record in the database from incoming audio bytes, storing its duration, score, language, and prompts. Takes audio and outputs the saved audio (now pointing at the stored record) plus a save result. Use it to persist freshly generated or imported clips as fresh records. Enable bulk import packs to batch many records into large storage packs for faster large imports, or mark records virtual."
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
                    style_prompt=audio.style_prompt,
                    voice_prompt=audio.voice_prompt,
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
    DESCRIPTION = "Overwrite the stored audio of an existing record with new bytes while keeping its name, segments, and other fields. Takes audio referencing an existing record and outputs the updated audio plus a save result. Score, language, and prompts are refreshed from the incoming audio when present and otherwise left as they were. Use it to replace a record's waveform after processing steps like resampling or enhancement, without creating a new record."
    CATEGORY = "Audio"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "save_result": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        audios: list[Audio] = [inputs["audio"] for inputs in batch]
        assert_unique_audio_ids(audios, self.NODE_TYPE)
        with database_session() as session:
            items = audio_crud.get_audio_files_bulk(
                session,
                [audio.audio_file_id for audio in audios],
            )
            payloads = {}
            for audio in audios:
                context.check_cancel()
                assert audio.data is not None, f"audio bytes are required: {audio.id}"
                item = items[audio.audio_file_id]
                payloads[audio.audio_file_id] = AudioUpdate(
                    name=item.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    score=_audio_score(audio, fallback=item.score),
                    language=_audio_language(audio, fallback=item.language),
                    style_prompt=audio.style_prompt if audio.style_prompt is not None else item.style_prompt,
                    voice_prompt=audio.voice_prompt if audio.voice_prompt is not None else item.voice_prompt,
                    segments=item.segments,
                    metadata=_audio_metadata(audio),
                    virtual=item.virtual,
                )
            updated_by_id = audio_crud.bulk_update_audio_files(session, payloads)
        return [
            _audio_writeback_output(updated_by_id[audio.audio_file_id], audio, "updated")
            for audio in context.cancellable(audios)
        ]


class LoadAudioSegmentsNode(Node):
    NODE_TYPE = "LoadAudioSegments"
    DESCRIPTION = "Load the saved segments for each audio record from the database and attach them to the passing audio. Takes audio referencing stored records and outputs the same audio with its segments replaced by whatever is persisted. Use it to bring existing transcripts, timings, and speaker labels back into a pipeline before further processing or splitting."
    CATEGORY = "Audio"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        refs = [_audio_ref_from_audio(inputs["audio"]) for inputs in batch]
        with database_session() as session:
            context.check_cancel()
            segments_by_id = audio_crud.list_audio_segments_bulk(
                session, [ref.audio_file_id for ref in refs]
            )
        outputs = []
        for inputs, ref in zip(batch, refs, strict=True):
            context.check_cancel()
            audio: Audio = inputs["audio"]
            segments = segments_by_id.get(ref.audio_file_id, [])
            audio_segments = [_audio_segment_from_dict(ref, segment) for segment in segments]
            outputs.append({"audio": replace(audio, segments=audio_segments)})
        return outputs


class SaveAudioSegmentsNode(Node):
    NODE_TYPE = "SaveAudioSegments"
    DESCRIPTION = "Persist an audio record's segments (transcript, timings, speaker, and alignment) to the database. Takes audio with segments and outputs the same audio with the saved segments plus a save result. In replace mode the record's segments are overwritten; in append mode the new segments are added alongside existing ones and re-sorted. Use it to store transcription or alignment results back onto a record."
    CATEGORY = "Audio"
    SETTINGS = SaveAudioSegmentsSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "save_result": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        audios: list[Audio] = [inputs["audio"] for inputs in batch]
        assert_unique_audio_ids(audios, self.NODE_TYPE)
        records: list[tuple[Audio, AudioRecordRef, SegmentGroup, list[dict[str, Any]]]] = []
        with database_session() as session:
            existing_by_id: dict[UUID, list[dict[str, Any]]] = {}
            if self.settings.mode == "append":
                existing_by_id = audio_crud.list_audio_segments_bulk(
                    session,
                    [_audio_ref_from_audio(inputs["audio"]).audio_file_id for inputs in batch],
                )
            for inputs in batch:
                context.check_cancel()
                audio: Audio = inputs["audio"]
                ref = _audio_ref_from_audio(audio)
                group = _segment_group_from_audio(audio)
                existing = existing_by_id.get(ref.audio_file_id, [])
                records.append((audio, ref, group, _new_group_segments(group, self.settings.mode, existing)))
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


def assert_unique_audio_ids(audios: list[Audio], operation: str) -> None:
    ids = [audio.audio_file_id for audio in audios]
    duplicates = sorted(str(audio_id) for audio_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{operation} received duplicate audio ids: {duplicates}")


def _audio_metadata(audio: Audio) -> dict[str, Any]:
    return {**audio.metadata, "sample_rate": audio.sample_rate, "channels": audio.channels}


def _audio_pack_config(settings: SaveAudioRecordSettings) -> audio_crud.AudioPackConfig:
    if not settings.bulk_import_packs:
        return audio_crud.AudioPackConfig()
    return audio_crud.AudioPackConfig(
        target_pack_bytes=512 * 1024 * 1024,
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
        style_prompt=item.style_prompt,
        voice_prompt=item.voice_prompt,
    )
    return {
        "audio": audio,
        "save_result": _save_result(f"db/audio/{item.id}", "audio_record", source.lineage_id, {"action": action}),
    }
