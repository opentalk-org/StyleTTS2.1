from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from runner.nodes.models import Audio, AudioSegment, stable_id


def audio_with_transcript_segments(
    model_name: str,
    audio: Audio,
    spans: list[tuple[float, float, str, float | None]],
    language: str,
    alignments: list[list[dict[str, Any]] | None] | None = None,
) -> Audio:
    filtered = [
        (
            audio.start + start,
            audio.start + end,
            text,
            confidence,
            alignments[index] if alignments is not None else None,
        )
        for index, (start, end, raw_text, confidence) in enumerate(spans)
        if (text := _clean_transcript_text(raw_text))
    ]
    text = " ".join(
        item_text.strip()
        for _start, _end, item_text, _confidence, _words in filtered
        if item_text.strip()
    ).strip()
    transcript_id = stable_id("transcript", model_name, audio.audio_file_id, audio.id)
    speaker = _diarized_speaker(audio)
    segments = [
        AudioSegment(
            source_audio_id=audio.audio_file_id,
            name=f"{model_name}:{audio.name}",
            start=start,
            end=end,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            text=item_text,
            phon="",
            id=stable_id("segment", transcript_id, index),
            lineage_id=stable_id("segment_lineage", audio.lineage_id, transcript_id, index),
            segment_id=stable_id("transcript_segment", transcript_id, index),
            speaker=speaker,
            confidence=confidence,
            alignment=_offset_alignment(words, audio.start),
            metadata={
                "transcript_id": transcript_id,
                "transcript_segment_index": index,
                "model": model_name,
                "type_": model_name,
            },
        )
        for index, (start, end, item_text, confidence, words) in enumerate(filtered)
    ]
    return replace(
        audio,
        segments=segments,
        metadata={
            **audio.metadata,
            "language": language,
            "model": model_name,
            "type_": model_name,
            "transcript_id": transcript_id,
            "transcript_text": text,
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
        },
    )


def _clean_transcript_text(text: str) -> str:
    cleaned = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return cleaned if cleaned.strip(". \t\r\n") else ""


def _offset_alignment(
    words: list[dict[str, Any]] | None,
    offset: float,
) -> list[dict[str, Any]] | None:
    if not words:
        return None
    return [
        {
            "word": word["word"],
            "start": word["start"] + offset,
            "end": word["end"] + offset,
        }
        for word in words
    ]


def _diarized_speaker(audio: Audio) -> str | None:
    if "diarization" not in audio.metadata:
        return None
    speaker = audio.metadata.get("speaker")
    return str(speaker) if speaker else None
