from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.asr.audio import extract_wav_range, wav_info
from runner.nodes.audio_segments.writeback_helpers import audio_segment_from_dict
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, AudioRecordRef, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.segment_references_crud import SegmentCursor, SegmentReference


PAGE_SIZE = 1_024


class SpeakerSegmentSourceSettings(StrictSettings):
    dataset_id: UUID


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
            audio_ids = list(dict.fromkeys(reference.audio_file_id for reference in references))
            stored = audio_crud.bulk_read_audio_files(session, audio_ids)

        outputs = []
        for reference in references:
            context.check_cancel()
            outputs.append({"audio": _segment_audio(reference, stored[reference.audio_file_id], self._segment_count)})
        self._after = references[-1].cursor
        self._emitted += len(outputs)
        await context.report_progress(
            self.id,
            self._emitted,
            self._segment_count,
            f"streamed {self._emitted}/{self._segment_count} stored segments",
        )
        return outputs


def _segment_audio(reference: SegmentReference, wav_bytes: bytes, source_count: int) -> Audio:
    source_ref = AudioRecordRef(
        audio_file_id=reference.audio_file_id,
        name=reference.audio_name,
        duration=reference.audio_duration,
        byte_length=reference.audio_byte_length,
        virtual=reference.audio_virtual,
        metadata=reference.audio_metadata,
    )
    stored_segment = audio_segment_from_dict(source_ref, reference.segment)
    info = wav_info(wav_bytes)
    clip_bytes = extract_wav_range(wav_bytes, stored_segment.start, stored_segment.end, info)
    duration = stored_segment.duration
    segment = replace(
        stored_segment,
        name=f"segment:{reference.audio_name}",
        start=0.0,
        end=duration,
        metadata={
            **stored_segment.metadata,
            "source_start": stored_segment.start,
            "source_end": stored_segment.end,
            "source_segment_index": reference.segment_index,
        },
    )
    clip_id = stable_id("speaker_segment_audio", reference.audio_file_id, segment.segment_id)
    return Audio(
        audio_file_id=reference.audio_file_id,
        name=segment.name,
        data=clip_bytes,
        sample_rate=int(info["sample_rate"]),
        channels=int(info["channels"]),
        start=0.0,
        end=duration,
        confidence=segment.confidence if segment.confidence is not None else 1.0,
        id=clip_id,
        lineage_id=segment.lineage_id,
        metadata={
            **reference.audio_metadata,
            "source_audio_id": str(reference.audio_file_id),
            "source_segment_id": segment.segment_id,
            "source_segment_index": reference.segment_index,
            "source_segment_count": source_count,
        },
        byte_length=len(clip_bytes),
        virtual=reference.audio_virtual,
        style_prompt=reference.style_prompt,
        voice_prompt=reference.voice_prompt,
        segments=[segment],
    )
