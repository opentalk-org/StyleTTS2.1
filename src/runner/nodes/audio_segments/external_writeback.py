from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from runflow.core.node import Node
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, SaveResultPort
from runner.nodes.models import Audio, AudioSegment, SaveResult, stable_id
from shared.db import database_session
from shared.db.audio.external_crud import bulk_create_external_audio_files
from shared.db.audio.schemas import ExternalAudioCreate, ExternalAudioLocation


class SaveExternalAudioRecordNode(Node):
    NODE_TYPE = "SaveExternalAudioRecord"
    DESCRIPTION = "Store metadata and transcript segments for external audio without copying audio bytes into object storage."
    CATEGORY = "Audio"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(), "save_result": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=256, max_size=512, timeout_ms=10)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 512

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio | SaveResult]]:
        audios = [inputs["audio"] for inputs in batch]
        payloads = []
        for audio in context.cancellable(audios):
            if audio.data is not None:
                raise ValueError(f"external audio record must not contain bytes: {audio.id}")
            payloads.append(_external_payload(audio))
        with database_session() as session:
            inserted = bulk_create_external_audio_files(session, payloads)
        await context.report_progress(
            self.id,
            len(audios),
            len(audios),
            f"stored {inserted} metadata records; skipped {len(audios) - inserted} existing",
        )
        return [_output(audio.audio_file_id, audio) for audio in audios]


def _external_payload(audio: Audio) -> ExternalAudioCreate:
    metadata = audio.metadata
    return ExternalAudioCreate(
        id=audio.audio_file_id,
        name=audio.name,
        duration=audio.duration,
        score=_optional_float(metadata["mos_score"]),
        language=str(metadata["language"]) if "language" in metadata and metadata["language"] else None,
        segments=[_segment_dict(segment) for segment in audio.segments],
        metadata=metadata,
        storage_ref=ExternalAudioLocation(
            provider=str(metadata["storage_provider"]),
            host=str(metadata["source_host"]),
            path=str(metadata["source_parquet_path"]),
            item_index=int(metadata["source_row_index"]),
        ),
    )


def _segment_dict(segment: AudioSegment) -> dict[str, Any]:
    type_ = str(segment.metadata["type_"])
    return {
        "id": segment.segment_id or segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "speaker": segment.speaker or "",
        "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
        "confidence": segment.confidence,
        "type_": type_,
        "metadata": segment.metadata,
        "alignment": segment.alignment,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _output(audio_file_id: UUID, audio: Audio) -> dict[str, Audio | SaveResult]:
    saved = replace(audio, audio_file_id=audio_file_id, virtual=True, byte_length=0)
    path = f"db/audio/{audio_file_id}"
    return {
        "audio": saved,
        "save_result": SaveResult(Path(path), "external_audio_record", stable_id("save", path), audio.lineage_id),
    }
