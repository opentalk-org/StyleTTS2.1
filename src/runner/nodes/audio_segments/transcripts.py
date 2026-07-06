from __future__ import annotations

from typing import Any, Literal

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import SEGMENT_GROUP, TRANSCRIPT
from runner.nodes.models import AudioSegment, SegmentGroup, Transcript, stable_id


class ApplyTranscriptSettings(StrictSettings):
    mode: Literal["replace", "add"] = "replace"


class TranscriptToSegmentsNode(Node):
    NODE_TYPE = "TranscriptToSegments"
    CATEGORY = "Audio / Segments"
    INPUTS = {"transcript": Port("transcript", TRANSCRIPT)}
    OUTPUTS = {"segment_group": Port("segment_group", SEGMENT_GROUP)}

    async def execute(self, batch, context):
        return [{"segment_group": _segment_group_from_transcript(inputs["transcript"])} for inputs in batch]


class ApplyTranscriptToSegmentsNode(Node):
    NODE_TYPE = "ApplyTranscriptToSegments"
    CATEGORY = "Audio / Segments"
    SETTINGS = ApplyTranscriptSettings
    INPUTS = {"transcript": Port("transcript", TRANSCRIPT), "segment_group": Port("segment_group", SEGMENT_GROUP)}
    OUTPUTS = {"segment_group": Port("segment_group", SEGMENT_GROUP)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            transcript: Transcript = inputs["transcript"]
            existing: SegmentGroup = inputs["segment_group"]
            converted = _segment_group_from_transcript(transcript)
            if self.settings.mode == "replace":
                outputs.append({
                    "segment_group": SegmentGroup(
                        converted.name,
                        converted.segments,
                        converted.id,
                        converted.lineage_id,
                        {**converted.metadata, "mode": self.settings.mode},
                    ),
                })
            else:
                segments = [*existing.segments, *converted.segments]
                group_id = stable_id("segment_group", self.settings.mode, existing.id, converted.id)
                metadata = {**existing.metadata, **converted.metadata, "mode": self.settings.mode}
                outputs.append({
                    "segment_group": SegmentGroup(existing.name, segments, group_id, converted.lineage_id, metadata),
                })
        return outputs


def _segment_group_from_transcript(transcript: Transcript) -> SegmentGroup:
    spans = transcript.segments or [_fallback_span(transcript)]
    segments = [_audio_segment_from_span(transcript, span, index) for index, span in enumerate(spans)]
    group_id = stable_id("segment_group", transcript.id, *(segment.id for segment in segments))
    metadata = {**transcript.metadata, "transcript_id": transcript.id, "model": transcript.model}
    return SegmentGroup(_segment_group_name(transcript), segments, group_id, transcript.lineage_id, metadata)


def _fallback_span(transcript: Transcript) -> dict[str, Any]:
    assert transcript.start is not None, f"transcript start is required when no spans exist: {transcript.id}"
    assert transcript.end is not None, f"transcript end is required when no spans exist: {transcript.id}"
    return {
        "id": stable_id("transcript_segment", transcript.id, transcript.start, transcript.end, 0),
        "start": transcript.start,
        "end": transcript.end,
        "text": transcript.text,
        "phon": "",
        "speaker": transcript.speaker or "",
    }


def _audio_segment_from_span(transcript: Transcript, span: dict[str, Any], index: int) -> AudioSegment:
    segment_id = str(span["id"]) if "id" in span else stable_id("transcript_segment", transcript.id, index)
    assert span["start"] is not None, f"span start is required: {segment_id}"
    assert span["end"] is not None, f"span end is required: {segment_id}"
    start = float(span["start"])
    end = float(span["end"])
    text = str(span["text"])
    phon = str(span["phon"]) if "phon" in span else ""
    speaker = str(span["speaker"]) if "speaker" in span and span["speaker"] else transcript.speaker
    return AudioSegment(
        source_audio_id=transcript.source_audio_id,
        name=_segment_group_name(transcript),
        start=start,
        end=end,
        sample_rate=_metadata_int(transcript, "sample_rate"),
        channels=_metadata_int(transcript, "channels"),
        text=text,
        phon=phon,
        id=stable_id("segment", transcript.id, segment_id, index),
        lineage_id=stable_id("segment_lineage", transcript.lineage_id, segment_id, index),
        segment_id=segment_id,
        speaker=speaker,
        metadata={"transcript_id": transcript.id, "transcript_segment_index": index, "model": transcript.model},
    )


def _segment_group_name(transcript: Transcript) -> str:
    return f"{transcript.model}:{transcript.source_audio_id}"


def _metadata_int(transcript: Transcript, key: str) -> int:
    assert key in transcript.metadata, f"transcript metadata missing {key}: {transcript.id}"
    return int(transcript.metadata[key])
