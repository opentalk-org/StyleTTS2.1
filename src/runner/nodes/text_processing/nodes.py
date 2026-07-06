from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import SEGMENT_GROUP, TRANSCRIPT
from runner.nodes.models import AudioSegment, SegmentGroup, Transcript, stable_id
from runner.nodes.text_runtime.phonemize import DEFAULT_PUNCTUATION_MARKS, phonemize_texts


class PhonemizeSettings(StrictSettings):
    language: str = "pl"
    tie: bool = True
    punctuation_marks: str = Field(default=DEFAULT_PUNCTUATION_MARKS, min_length=1, max_length=512)
    espeak_workers: int = Field(default=4, ge=1, le=64)
    align_threads: int = Field(default=8, ge=1, le=64)


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
            phonemes = _phonemize_texts([str(span["text"]) for span in spans], self.settings)
            segments = [_phonemized_span(span, phon) for span, phon in zip(spans, phonemes, strict=True)]
            metadata = {
                **transcript.metadata,
                "phoneme_language": self.settings.language,
                "tie": self.settings.tie,
                "punctuation_marks": self.settings.punctuation_marks,
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
            segments = _phonemized_segments(group.segments, self.settings)
            metadata = {
                **group.metadata,
                "phoneme_language": self.settings.language,
                "tie": self.settings.tie,
                "punctuation_marks": self.settings.punctuation_marks,
                "phoneme_mode": self.settings.mode,
            }
            group_id = stable_id("segment_group_phon", group.id, self.settings.language, self.settings.mode)
            outputs.append({
                "segment_group": SegmentGroup(group.name, segments, group_id, group.lineage_id, metadata),
            })
        return outputs


def _phonemized_span(span: dict[str, Any], phonemes: str) -> dict[str, Any]:
    return {**span, "phon": phonemes}


def _fallback_span(transcript: Transcript) -> dict[str, Any]:
    return {
        "id": stable_id("transcript_segment", transcript.id, transcript.start, transcript.end, 0),
        "start": transcript.start,
        "end": transcript.end,
        "text": transcript.text,
        "phon": "",
        "speaker": transcript.speaker or "",
    }


def _phonemized_segments(segments: list[AudioSegment], settings: PhonemizeSegmentsSettings) -> list[AudioSegment]:
    work_segments = [segment for segment in segments if _should_phonemize_segment(segment, settings)]
    phonemes = _phonemize_texts([segment.text for segment in work_segments], settings)
    phoneme_iter = iter(phonemes)
    return [
        replace(segment, phon=next(phoneme_iter))
        if _should_phonemize_segment(segment, settings)
        else segment
        for segment in segments
    ]


def _should_phonemize_segment(segment: AudioSegment, settings: PhonemizeSegmentsSettings) -> bool:
    return bool(segment.text.strip()) and (settings.mode == "replace" or not segment.phon.strip())


def _phonemize_texts(texts: list[str], settings: PhonemizeSettings) -> list[str]:
    return phonemize_texts(
        texts,
        language=settings.language,
        tie=settings.tie,
        punctuation_marks=settings.punctuation_marks,
        espeak_workers=settings.espeak_workers,
        align_threads=settings.align_threads,
    )
