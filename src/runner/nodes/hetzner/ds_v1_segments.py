from __future__ import annotations

import json
import math
from typing import Any

from runner.nodes.hetzner.ds_v1_metadata import DsV2Sample
from runner.nodes.hetzner.ds_v2_alignment import alignment_from_timestamps, alignment_window
from runner.nodes.hetzner.ds_v2_audio import TRANSCRIPT_SEGMENTS
from runner.nodes.models import Audio, AudioSegment, stable_id
from shared.audio_annotations import AudioAnnotations


def segments_from_samples(
    audio: Audio,
    samples: tuple[DsV2Sample, ...],
    remote_v1_path: str,
    row_index: int,
    preferred_text_column: str,
) -> list[AudioSegment]:
    ordered = sorted(
        samples,
        key=lambda sample: (
            _float(sample, "sample_start"),
            _int(sample, "sample_index"),
            sample.row_index,
        ),
    )
    segments = []
    for sample in ordered:
        segments.extend(
            _sample_segments(audio, sample, remote_v1_path, row_index, preferred_text_column)
        )
    return segments


def _sample_segments(
    audio: Audio,
    sample: DsV2Sample,
    remote_v1_path: str,
    row_index: int,
    preferred_text_column: str,
) -> list[AudioSegment]:
    start, sample_end, end = _segment_bounds(audio, sample)
    score = _optional_float(sample, "mos_score")
    speaker_id = _optional_text(sample, "speaker_id")
    timestamps = _json_value(sample, "text_timestamps")
    source_metadata = _json_value(sample, "metadata")
    if not isinstance(source_metadata, dict):
        raise ValueError(f"ds_v2 metadata row {sample.row_index} metadata must be an object")
    segments = []
    for source, column in TRANSCRIPT_SEGMENTS:
        text = _optional_text(sample, column)
        if text is None:
            continue
        alignment = None
        if column == "text_parakeet" and timestamps is not None:
            local = alignment_from_timestamps(
                timestamps,
                text,
                alignment_window(sample.values, sample.row_index),
            )
            alignment = _absolute_alignment(local, start, sample_end, sample.row_index)
        segment_key = (remote_v1_path, row_index, sample.row_index, source)
        segments.append(
            AudioSegment(
                source_audio_id=audio.audio_file_id,
                name=audio.name,
                start=start,
                end=end,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                text=text,
                phon="",
                id=stable_id("hetzner_ds_v1_segment", *segment_key),
                lineage_id=stable_id("hetzner_ds_v1_segment_lineage", *segment_key),
                segment_id=stable_id("hetzner_ds_v1_segment_entry", *segment_key),
                annotations=AudioAnnotations(
                    speaker_id=speaker_id,
                    score=score,
                    metadata=_segment_metadata(
                    sample,
                    source,
                    column,
                    preferred_text_column,
                    timestamps,
                    source_metadata,
                    ),
                ),
                alignment=alignment,
            )
        )
    return segments


def _segment_bounds(audio: Audio, sample: DsV2Sample) -> tuple[float, float, float]:
    start = _float(sample, "sample_start")
    sample_end = _float(sample, "sample_end")
    duration = _float(sample, "duration")
    values = (start, sample_end, duration, audio.duration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"ds_v2 metadata row {sample.row_index} has non-finite timing")
    if start < 0 or sample_end < start or sample_end > audio.duration:
        raise ValueError(
            f"ds_v2 metadata row {sample.row_index} sample window {start}..{sample_end} "
            f"is outside recording duration {audio.duration}"
        )
    if duration <= 0:
        raise ValueError(f"ds_v2 metadata row {sample.row_index} has invalid duration {duration}")
    return start, sample_end, min(audio.duration, start + duration)


def _absolute_alignment(
    local: list[dict[str, Any]] | None,
    sample_start: float,
    sample_end: float,
    row_index: int,
) -> list[dict[str, Any]] | None:
    if local is None:
        return None
    alignment = [
        {
            **entry,
            "start": _snap_to_window(sample_start + float(entry["start"]), sample_start, sample_end),
            "end": _snap_to_window(sample_start + float(entry["end"]), sample_start, sample_end),
        }
        for entry in local
    ]
    previous_start = sample_start
    for entry in alignment:
        start = float(entry["start"])
        end = float(entry["end"])
        if not sample_start <= start <= end <= sample_end or start < previous_start:
            raise ValueError(f"ds_v2 metadata row {row_index} produced invalid absolute alignment")
        previous_start = start
    return alignment


def _snap_to_window(value: float, start: float, end: float) -> float:
    if math.isclose(value, start, rel_tol=0.0, abs_tol=1e-6):
        return start
    if math.isclose(value, end, rel_tol=0.0, abs_tol=1e-6):
        return end
    return value


def _segment_metadata(
    sample: DsV2Sample,
    source: str,
    column: str,
    preferred_text_column: str,
    timestamps: Any,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type_": source,
        "model": source,
        "text_column": column,
        "preferred_text_column": preferred_text_column,
        "text_timestamps": timestamps,
        "source_metadata": source_metadata,
        "ds_v2_metadata_row_index": sample.row_index,
        "duration": _float(sample, "duration"),
        "chunk_index": _int(sample, "chunk_index"),
        "chunk_start": _float(sample, "chunk_start"),
        "chunk_end": _float(sample, "chunk_end"),
        "speaker_start": _float(sample, "speaker_start"),
        "speaker_end": _float(sample, "speaker_end"),
        "sample_index": _int(sample, "sample_index"),
        "sample_start": _float(sample, "sample_start"),
        "sample_end": _float(sample, "sample_end"),
        "parquet_filename": _optional_text(sample, "parquet_filename"),
        "src_type": _optional_text(sample, "src_type"),
        "src": _optional_text(sample, "src"),
    }


def _json_value(sample: DsV2Sample, key: str) -> Any:
    text = _optional_text(sample, key)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"ds_v2 metadata row {sample.row_index} has invalid {key} JSON") from error


def _float(sample: DsV2Sample, key: str) -> float:
    try:
        return float(sample.values[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ds_v2 metadata row {sample.row_index} has invalid {key}") from error


def _optional_float(sample: DsV2Sample, key: str) -> float | None:
    text = _optional_text(sample, key)
    return float(text) if text is not None else None


def _int(sample: DsV2Sample, key: str) -> int:
    try:
        return int(sample.values[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ds_v2 metadata row {sample.row_index} has invalid {key}") from error


def _optional_text(sample: DsV2Sample, key: str) -> str | None:
    text = sample.values[key].strip()
    return text or None
