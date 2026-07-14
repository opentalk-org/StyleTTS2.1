from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentWindow:
    source_start: float
    source_end: float

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


def alignment_window(row: dict[str, Any], row_index: int) -> AlignmentWindow:
    try:
        chunk_start = float(row["chunk_start"])
        sample_start = float(row["sample_start"])
        sample_end = float(row["sample_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ds_v2 row {row_index} has incomplete alignment window") from error
    if not all(math.isfinite(value) for value in (chunk_start, sample_start, sample_end)):
        raise ValueError(f"ds_v2 row {row_index} has non-finite alignment window")
    if sample_start < chunk_start or sample_end < sample_start:
        raise ValueError(
            f"ds_v2 row {row_index} has invalid alignment window: "
            f"chunk_start={chunk_start}, sample_start={sample_start}, sample_end={sample_end}"
        )
    return AlignmentWindow(sample_start - chunk_start, sample_end - chunk_start)


def alignment_from_timestamps(
    timestamps: Any,
    text: str,
    window: AlignmentWindow,
) -> list[dict[str, Any]] | None:
    if isinstance(timestamps, dict):
        timestamps = timestamps["word"]
    if not isinstance(timestamps, list):
        return None
    transcript_words = text.split()
    if not transcript_words:
        return None
    timestamp_words = [str(item["word"]).strip() for item in timestamps]
    width = len(transcript_words)
    candidates = [
        start
        for start in range(len(timestamps) - width + 1)
        if timestamp_words[start:start + width] == transcript_words
    ]
    if not candidates:
        return None
    selected_start = min(
        candidates,
        key=lambda start: (
            abs(float(timestamps[start]["start"]) - window.source_start)
            + abs(float(timestamps[start + width - 1]["end"]) - window.source_end),
            start,
        ),
    )
    alignment = []
    for item in timestamps[selected_start:selected_start + width]:
        word = str(item["word"]).strip()
        start = min(window.duration, max(0.0, float(item["start"]) - window.source_start))
        end = min(window.duration, max(0.0, float(item["end"]) - window.source_start))
        alignment.append({"word": word, "start": start, "end": max(start, end)})
    return alignment
