from __future__ import annotations

import io
import wave
from dataclasses import replace
from typing import Any, Literal
from uuid import UUID

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.audio_segments.extract_writeback import (
    metadata_segment_payloads as _metadata_segment_payloads,
    save_result as _save_result,
    saved_segment_group as _saved_segment_group,
)
from runner.nodes.audio_segments.persist_split import (
    persist_split_records,
)
from runner.nodes.datatypes import AudioPort, SaveResultPort
from runner.nodes.models import Audio, SegmentGroup, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.clickhouse.models import AudioFileRecord
from shared.db.audio.schemas import AudioCreate


class PersistSplitAudioRecordsSettings(StrictSettings):
    target_dataset_id: UUID | None = None
    source_dataset_id: UUID | None = None
    mode: Literal["create_new", "replace_all"] = "create_new"
    delete_source_on_replace: bool = False
    virtual: bool = False


class ExtractSegmentGroupAudioNode(Node):
    NODE_TYPE = "ExtractSegmentGroupAudio"
    DESCRIPTION = "Cut out the slice of the original recording that a planned segment group spans and emit it as a standalone audio clip. Takes planned-group audio (referencing a source recording and a span), reads the source WAV from storage, and outputs a new clip trimmed to the group's start and end with its segment timings shifted to be clip-local. Use it after planning segment groups to turn each group into an independent audio file ready to be saved."
    CATEGORY = "Audio"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.FULL_WINDOW)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        audios: list[Audio] = [inputs["audio"] for inputs in batch]
        groups = [_group_from_audio(audio) for audio in audios]
        source_ids = [_group_source_audio_id(group) for group in groups]
        with database_session() as session:
            items = audio_crud.get_audio_files_bulk(session, source_ids)
            stored = audio_crud.bulk_read_audio_files(session, source_ids)
        outputs = []
        for group, source_id in context.cancellable(
            zip(groups, source_ids, strict=True)
        ):
            source = _source_audio(items[source_id], stored[source_id], group)
            outputs.append({"audio": extract_group_audio(source, group)})
        return outputs


class PersistSplitAudioRecordsNode(Node):
    NODE_TYPE = "PersistSplitAudioRecords"
    DESCRIPTION = "Save each split audio clip as a new audio record in the database along with its segments, and optionally attach it to datasets. Takes an extracted clip and outputs the saved audio (now pointing at the stored record) plus a save result. In replace-all mode it can also detach or delete the original source record once all of its groups have been written. Use it as the final step of a split-and-persist pipeline; set the target and source datasets and whether records are virtual."
    CATEGORY = "Audio"
    SETTINGS = PersistSplitAudioRecordsSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {
        "audio": AudioPort(),
        "save_result": SaveResultPort(),
    }
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        audios: list[Audio] = [inputs["audio"] for inputs in batch]
        payloads = []
        for audio in context.cancellable(audios):
            assert audio.data is not None, f"audio bytes are required: {audio.id}"
            payloads.append(
                AudioCreate(
                    name=audio.name,
                    wav_bytes=audio.data,
                    duration=audio.duration,
                    annotations=audio.annotations.model_copy(
                        update={"metadata": _target_audio_metadata(audio)}
                    ),
                    language=_target_audio_language(audio),
                    style_prompt=audio.style_prompt,
                    voice_prompt=audio.voice_prompt,
                    segments=[],
                    virtual=self.settings.virtual,
                )
            )
        items, segments_by_id = persist_split_records(
            audios=audios,
            payloads=payloads,
            segment_payloads=[_metadata_segment_payloads(audio) for audio in audios],
            target_dataset_id=self.settings.target_dataset_id,
            source_dataset_id=self.settings.source_dataset_id,
            replace_all=self.settings.mode == "replace_all",
            delete_source=self.settings.delete_source_on_replace,
        )
        outputs = []
        for item, audio in zip(items, audios, strict=True):
            segments = segments_by_id[item.id]
            source_group_id = str(audio.metadata["source_group_id"])
            saved_group = _saved_segment_group(item, audio, segments)
            saved_audio = replace(
                audio,
                audio_file_id=item.id,
                name=item.name,
                id=stable_id("audio", item.id, item.name),
                lineage_id=stable_id("audio_ref", item.id),
                segments=saved_group.segments,
                annotations=audio_crud.audio_file_annotations(item),
                byte_length=item.byte_length,
                virtual=item.virtual,
            )
            outputs.append(
                {
                    "audio": saved_audio,
                    "save_result": _save_result(
                        item, audio.lineage_id, source_group_id, len(segments)
                    ),
                }
            )
        return outputs


def _group_from_audio(audio: Audio) -> SegmentGroup:
    return SegmentGroup(
        audio.name,
        audio.segments,
        stable_id(
            "segment_group", audio.id, *(segment.id for segment in audio.segments)
        ),
        audio.lineage_id,
        audio.metadata,
    )


def extract_group_audio(audio: Audio, group: SegmentGroup) -> Audio:
    info = _read_wav_info(audio.data)
    span_start, span_end = group_span_bounds(group)
    start_frame = _seconds_to_frame(span_start, info["sample_rate"])
    end_frame = _seconds_to_frame(span_end, info["sample_rate"])
    assert 0 <= start_frame < end_frame <= info["frame_count"], (
        f"group span outside audio bounds: {group.id}"
    )
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
        audio.annotations.model_copy(update={"metadata": metadata}),
        audio_id,
        group.lineage_id,
    )


def adjusted_segment_payloads(group: SegmentGroup) -> list[dict[str, Any]]:
    span_start, _span_end = group_span_bounds(group)
    payloads = []
    for index, segment in enumerate(group.segments):
        start = max(0.0, segment.start - span_start)
        end = max(start, segment.end - span_start)
        payloads.append(
            {
                "id": stable_id(
                    "segment", group.id, index, segment.id, segment.segment_id or ""
                ),
                "start": start,
                "end": end,
                "text": segment.text,
                "phon": segment.phon,
                "annotations": segment.annotations.model_copy(
                    update={
                        "metadata": {
                            **segment.metadata,
                            "type_": str(
                                segment.metadata.get(
                                    "type_", segment.metadata.get("model", "manual")
                                )
                            ),
                            "source_audio_id": str(segment.source_audio_id),
                            "source_segment_id": segment.segment_id or segment.id,
                            "source_segment_lineage_id": segment.lineage_id,
                        }
                    }
                ).model_dump(mode="json"),
                "type_": str(
                    segment.metadata.get(
                        "type_", segment.metadata.get("model", "manual")
                    )
                ),
            }
        )
    return payloads


def group_span_bounds(group: SegmentGroup) -> tuple[float, float]:
    assert group.segments, f"segment group is empty: {group.id}"
    return min(segment.start for segment in group.segments), max(
        segment.end for segment in group.segments
    )


def _group_source_audio_id(group: SegmentGroup) -> UUID:
    source_ids = {segment.source_audio_id for segment in group.segments}
    assert len(source_ids) == 1, f"group has multiple source audio ids: {group.id}"
    return next(iter(source_ids))


def _source_audio(item: AudioFileRecord, data: bytes, group: SegmentGroup) -> Audio:
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
        audio_crud.audio_file_annotations(item).model_copy(
            update={"metadata": metadata}
        ),
        audio_id,
        group.lineage_id,
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
        raise ValueError(
            "ExtractSegmentGroupAudio supports WAV audio bytes only"
        ) from exc


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


def _target_audio_language(audio: Audio) -> str | None:
    value = audio.metadata.get("language")
    if value is None or value == "":
        return None
    return str(value)
