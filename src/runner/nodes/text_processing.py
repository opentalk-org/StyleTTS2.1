from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import SEGMENT_GROUP, TRANSCRIPT
from runner.nodes.models import AudioSegment, SegmentGroup, Transcript, stable_id


class PhonemizeSettings(StrictSettings):
    language: str = "en-us"
    tie: bool = True
    punctuation: bool = True
    workers: int = Field(default=4, ge=1, le=32)
    threads: int = Field(default=2, ge=1, le=16)


class PhonemizeSegmentsSettings(PhonemizeSettings):
    mode: Literal["fill", "replace"] = "fill"


class PhonemizeTranscriptNode(Node):
    NODE_TYPE = "PhonemizeTranscript"
    CATEGORY = "Text"
    SETTINGS = PhonemizeSettings
    INPUTS = {"transcript": Port("transcript", TRANSCRIPT)}
    OUTPUTS = {"transcript": Port("transcript", TRANSCRIPT)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            transcript: Transcript = inputs["transcript"]
            spans = transcript.segments or [_fallback_span(transcript)]
            segments = [_phonemized_span(span, self.settings) for span in spans]
            metadata = {
                **transcript.metadata,
                "phoneme_language": self.settings.language,
                "tie": self.settings.tie,
                "punctuation": self.settings.punctuation,
            }
            outputs.append({
                "transcript": Transcript(
                    transcript.text,
                    transcript.model,
                    transcript.source_audio_id,
                    transcript.start,
                    transcript.end,
                    transcript.speaker,
                    stable_id("phon", transcript.id, self.settings.language),
                    transcript.lineage_id,
                    segments,
                    metadata,
                )
            })
        return outputs


class PhonemizeSegmentsNode(Node):
    NODE_TYPE = "PhonemizeSegments"
    CATEGORY = "Text"
    SETTINGS = PhonemizeSegmentsSettings
    INPUTS = {"segment_group": Port("segment_group", SEGMENT_GROUP)}
    OUTPUTS = {"segment_group": Port("segment_group", SEGMENT_GROUP)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            group: SegmentGroup = inputs["segment_group"]
            segments = [_phonemized_segment(segment, self.settings) for segment in group.segments]
            metadata = {
                **group.metadata,
                "phoneme_language": self.settings.language,
                "tie": self.settings.tie,
                "punctuation": self.settings.punctuation,
                "phoneme_mode": self.settings.mode,
            }
            group_id = stable_id("segment_group_phon", group.id, self.settings.language, self.settings.mode)
            outputs.append({
                "segment_group": SegmentGroup(group.name, segments, group_id, group.lineage_id, metadata),
            })
        return outputs


def _phonemized_span(span: dict[str, Any], settings: PhonemizeSettings) -> dict[str, Any]:
    text = str(span["text"])
    return {**span, "phon": _placeholder_phonemes(text, settings)}


def _fallback_span(transcript: Transcript) -> dict[str, Any]:
    return {
        "id": stable_id("transcript_segment", transcript.id, transcript.start, transcript.end, 0),
        "start": transcript.start,
        "end": transcript.end,
        "text": transcript.text,
        "phon": "",
        "speaker": transcript.speaker or "",
    }


def _phonemized_segment(segment: AudioSegment, settings: PhonemizeSegmentsSettings) -> AudioSegment:
    if settings.mode == "fill" and segment.phon:
        return segment
    return replace(segment, phon=_placeholder_phonemes(segment.text, settings))


def _placeholder_phonemes(text: str, settings: PhonemizeSettings) -> str:
    units = [character.lower() for character in text if settings.punctuation or character.isalnum() or character.isspace()]
    separator = "\u0361" if settings.tie else " "
    return separator.join(character for character in units if not character.isspace())
