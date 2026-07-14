from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


TIMESTAMP_ABS_TOLERANCE = 1e-9


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
    window: AlignmentWindow,
) -> list[dict[str, Any]] | None:
    if isinstance(timestamps, dict):
        timestamps = timestamps["word"]
    if not isinstance(timestamps, list):
        return None
    alignment = []
    for item in timestamps:
        word = str(item["word"]).strip()
        source_start = float(item["start"])
        source_end = float(item["end"])
        starts_at_end = source_start >= window.source_end or math.isclose(
            source_start, window.source_end, rel_tol=0.0, abs_tol=TIMESTAMP_ABS_TOLERANCE
        )
        ends_at_start = source_end <= window.source_start or math.isclose(
            source_end, window.source_start, rel_tol=0.0, abs_tol=TIMESTAMP_ABS_TOLERANCE
        )
        if not word or starts_at_end or ends_at_start:
            continue
        start = max(0.0, source_start - window.source_start)
        end = min(window.duration, source_end - window.source_start)
        alignment.append({"word": word, "start": start, "end": max(start, end)})
    return alignment or None


def validate_alignment_text(
    text: str,
    alignment: list[dict[str, Any]] | None,
    row_index: int,
) -> None:
    transcript = " ".join(text.split())
    aligned = " ".join(str(entry["word"]).strip() for entry in alignment or [])
    if aligned != transcript:
        raise ValueError(
            f"ds_v2 row {row_index} Parakeet alignment does not match transcript: "
            f"transcript={transcript!r}, alignment={aligned!r}"
        )
