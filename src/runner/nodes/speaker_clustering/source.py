from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.audio_segments.writeback_helpers import audio_segment_from_dict
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, AudioRecordRef, AudioSegment, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.ranges import (
    SegmentReadRequest,
    WavClip,
    bulk_read_wav_segments,
)
from shared.db.audio.segment_catalog import SegmentCursor, SegmentReference


PAGE_SIZE = 1_024


class SpeakerSegmentSourceSettings(StrictSettings):
    dataset_id: UUID
    maximum_page_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    audio_fetch_workers: int = Field(default=16, gt=0)


class SpeakerSegmentSource(Node):
    NODE_TYPE = "SpeakerSegmentSource"
    DESCRIPTION = "Stream stored dataset segments as bounded, one-segment audio clips for speaker embedding."
    CATEGORY = "Speaker Clustering"
    SETTINGS = SpeakerSegmentSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = PAGE_SIZE

    def __init__(self, node_id: str | None = None, **params: Any) -> None:
        super().__init__(node_id=node_id, **params)
        self._after: SegmentCursor | None = None
        self._emitted = 0
        with database_session() as session:
            self._segment_count = audio_crud.count_segment_references(
                session,
                self.settings.dataset_id,
            )

    def remaining_items(self, context: Any) -> int:
        return self._segment_count - self._emitted

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        assert len(batch) == 1, f"{self.id} requires one source task"
        context.check_cancel()
        limit = min(PAGE_SIZE, self.runtime.queue_max_size, self.remaining_items(context))
        with database_session() as session:
            references = audio_crud.list_segment_references_page(
                session,
                self.settings.dataset_id,
                self._after,
                limit,
            )
            assert references, f"{self.id} expected {self.remaining_items(context)} more database segments"
            references = bounded_clip_prefix(
                references, self.settings.maximum_page_bytes
            )
            stored_segments = [_stored_segment(reference) for reference in references]
            clips = bulk_read_wav_segments(
                session,
                [
                    SegmentReadRequest(
                        reference.audio_file_id,
                        segment.start,
                        segment.end,
                    )
                    for reference, segment in zip(
                        references, stored_segments, strict=True
                    )
                ],
                self.settings.audio_fetch_workers,
            )

        outputs = []
        for reference, stored_segment, clip in zip(
            references, stored_segments, clips, strict=True
        ):
            context.check_cancel()
            outputs.append(
                {
                    "audio": _segment_audio(
                        reference,
                        stored_segment,
                        clip,
                        self._segment_count,
                        self.settings.dataset_id,
                    )
                }
            )
        self._after = references[-1].cursor
        self._emitted += len(outputs)
        await context.report_progress(
            self.id,
            self._emitted,
            self._segment_count,
            f"streamed {self._emitted}/{self._segment_count} stored segments",
        )
        return outputs


def bounded_clip_prefix(
    references: list[SegmentReference], maximum_bytes: int
) -> list[SegmentReference]:
    selected = []
    output_bytes = 0
    for reference in references:
        clip_bytes = _estimated_clip_bytes(reference)
        if selected and output_bytes + clip_bytes > maximum_bytes:
            break
        selected.append(reference)
        output_bytes += clip_bytes
    return selected


def _estimated_clip_bytes(reference: SegmentReference) -> int:
    start = float(reference.segment["start"])
    end = float(reference.segment["end"])
    duration = max(0.0, end - start)
    bytes_per_second = reference.audio_byte_length / reference.audio_duration
    return max(44, int(round(duration * bytes_per_second)))


def _stored_segment(reference: SegmentReference) -> AudioSegment:
    return audio_segment_from_dict(_source_ref(reference), reference.segment)


def _source_ref(reference: SegmentReference) -> AudioRecordRef:
    return AudioRecordRef(
        audio_file_id=reference.audio_file_id,
        name=reference.audio_name,
        duration=reference.audio_duration,
        byte_length=reference.audio_byte_length,
        virtual=reference.audio_virtual,
        annotations=reference.annotations,
    )


def _segment_audio(
    reference: SegmentReference,
    stored_segment: AudioSegment,
    clip: WavClip,
    source_count: int,
    dataset_id: UUID,
) -> Audio:
    duration = stored_segment.duration
    segment = replace(
        stored_segment,
        name=f"segment:{reference.audio_name}",
        start=0.0,
        end=duration,
        annotations=stored_segment.annotations.model_copy(update={"metadata": {
            **stored_segment.metadata,
            "source_start": stored_segment.start,
            "source_end": stored_segment.end,
            "source_segment_index": reference.segment_index,
        }}),
    )
    clip_id = stable_id("speaker_segment_audio", reference.audio_file_id, segment.segment_id)
    return Audio(
        audio_file_id=reference.audio_file_id,
        name=segment.name,
        data=clip.data,
        sample_rate=clip.sample_rate,
        channels=clip.channels,
        start=0.0,
        end=duration,
        annotations=segment.annotations.model_copy(update={"metadata": {
            **reference.annotations.metadata,
            "source_audio_id": str(reference.audio_file_id),
            "source_segment_id": segment.segment_id,
            "source_segment_index": reference.segment_index,
            "source_segment_count": source_count,
            "dataset_id": str(dataset_id),
        }}),
        id=clip_id,
        lineage_id=segment.lineage_id,
        byte_length=len(clip.data),
        virtual=reference.audio_virtual,
        style_prompt=reference.style_prompt,
        voice_prompt=reference.voice_prompt,
        segments=[segment],
    )
