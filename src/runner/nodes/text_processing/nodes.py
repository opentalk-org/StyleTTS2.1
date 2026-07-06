from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import AUDIO
from runner.nodes.models import Audio, AudioSegment
from runner.nodes.text.runtime.phonemize import DEFAULT_PUNCTUATION_MARKS, phonemize_texts


class PhonemizeSettings(StrictSettings):
    language: str = "pl"
    tie: bool = True
    punctuation_marks: str = Field(default=DEFAULT_PUNCTUATION_MARKS, min_length=1, max_length=512)
    espeak_workers: int = Field(default=4, ge=1, le=64)
    align_threads: int = Field(default=8, ge=1, le=64)


class PhonemizeSegmentsSettings(PhonemizeSettings):
    mode: Literal["fill", "replace"] = "fill"


class PhonemizeSegmentsNode(Node):
    NODE_TYPE = "PhonemizeSegments"
    CATEGORY = "Text"
    SETTINGS = PhonemizeSegmentsSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio: Audio = inputs["audio"]
            segments = _phonemized_segments(audio.segments, self.settings)
            metadata = {
                **audio.metadata,
                "phoneme_language": self.settings.language,
                "tie": self.settings.tie,
                "punctuation_marks": self.settings.punctuation_marks,
                "phoneme_mode": self.settings.mode,
            }
            outputs.append({"audio": replace(audio, segments=segments, metadata=metadata)})
        return outputs


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
