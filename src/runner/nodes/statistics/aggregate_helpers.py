import re
from collections import Counter
from math import isfinite
from typing import Any

from runner.nodes.statistics.smart_histogram import histogram_counts


BREAK_TAG_PATTERN = re.compile(r"<break t=(\d+)>")


def pooled_histogram(records: list[dict[str, Any]], field: str, bins: int, fallback: tuple[float, float], clip: bool = False) -> dict[str, Any]:
    values: list[float] = []
    for record in records:
        values.extend(float(value) for value in record[field] if isfinite(float(value)))
    value_range = fallback if clip else (_range_for(values) if values else fallback)
    return histogram_counts(values, bins, value_range)


def char_unigram_counts(text: str, limit: int = 80) -> list[list[Any]]:
    counts = Counter(char for char in text if not char.isspace())
    return [[char, int(count)] for char, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def char_bigram_matrix(text: str, max_labels: int = 28) -> dict[str, Any]:
    labels = [char for char, _ in Counter(char for char in text if not char.isspace()).most_common(max_labels)]
    index = {char: pos for pos, char in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for pos in range(max(0, len(text) - 1)):
        left, right = text[pos], text[pos + 1]
        if left in index and right in index:
            matrix[index[left]][index[right]] += 1
    return {"labels": labels, "matrix": matrix}


def char_trigram_extremes(text: str, top_n: int = 10) -> tuple[list[list[Any]], list[list[Any]]]:
    grams = [text[index : index + 3] for index in range(max(0, len(text) - 2)) if not any(char.isspace() for char in text[index : index + 3])]
    counts = Counter(grams)
    top = [[gram, int(count)] for gram, count in counts.most_common(top_n)]
    bottom = [[gram, int(count)] for gram, count in sorted(counts.items(), key=lambda item: (item[1], item[0]))[:top_n]]
    return top, bottom


def flatten_segments(items: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in items:
        record = record_value(item)
        if "segments" in record:
            flattened.extend(_segments_with_parent_audio_id(record))
            continue
        flattened.append(record)
    return flattened


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def record_value(item: Any) -> dict[str, Any]:
    assert isinstance(item, dict), f"statistics JSON record must be a dict, got {type(item).__name__}"
    return item


def _segments_with_parent_audio_id(record: dict[str, Any]) -> list[dict[str, Any]]:
    segments = record["segments"]
    assert isinstance(segments, list), "segments wrapper must contain a list"
    parent_audio_id = _parent_audio_id(record)
    out = []
    for item in segments:
        segment = record_value(item)
        if parent_audio_id is not None and _parent_audio_id(segment) is None:
            segment = {**segment, "source_audio_id": parent_audio_id}
        out.append(segment)
    return out


def _parent_audio_id(record: dict[str, Any]) -> str | None:
    for field in ("audio_file_id", "source_audio_id", "source_id"):
        if field in record:
            return str(record[field])
    return None


def _range_for(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if hi > lo:
        return lo, hi
    epsilon = 1e-9 if lo == 0 else abs(lo) * 1e-9
    return lo - epsilon, hi + epsilon


def finite_field_values(records: list[dict[str, Any]], field: str) -> list[float]:
    return [float(record[field]) for record in records if record[field] is not None and isfinite(float(record[field]))]


def break_histograms(
    file_ids: list[str],
    segments: list[dict[str, Any]],
    bins: int,
) -> dict[str, dict[str, Any]]:
    counts = {audio_id: 0 for audio_id in file_ids}
    durations: list[float] = []
    for segment in segments:
        values = [int(value) for value in BREAK_TAG_PATTERN.findall(str(segment["text"]))]
        counts[segment_audio_id(segment)] += len(values)
        durations.extend(float(value) for value in values)
    return {
        "break_count_per_file_histogram": histogram_counts(
            [float(counts[audio_id]) for audio_id in file_ids], bins
        ),
        "break_duration_ms_histogram": histogram_counts(durations, bins),
    }


def string_value(record: dict[str, Any], field: str) -> str:
    value = record[field]
    return "" if value is None else str(value)


def float_value(record: dict[str, Any], field: str) -> float:
    return float(record[field])


def segment_audio_id(segment: dict[str, Any]) -> str:
    for field in ("audio_file_id", "source_audio_id", "source_id"):
        if field in segment:
            return str(segment[field])
    raise KeyError("segment record missing audio_file_id/source_audio_id/source_id")


def segment_phon(segment: dict[str, Any]) -> str:
    for field in ("phon", "phonemes"):
        if field in segment:
            return "" if segment[field] is None else str(segment[field])
    return ""


def segment_speaker(segment: dict[str, Any]) -> str:
    if "speaker" in segment and str(segment["speaker"]).strip():
        return str(segment["speaker"]).strip()
    return "-"


def inter_word_silences(segment: dict[str, Any], max_seconds: float) -> list[float]:
    times = segment.get("word_times") or []
    ordered = sorted(times, key=lambda pair: float(pair[0]))
    gaps: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        gap = float(current[0]) - float(previous[1])
        gaps.append(min(max(gap, 0.0), max_seconds))
    return gaps


def segment_duration(segment: dict[str, Any]) -> float:
    if "duration" in segment:
        return max(0.0, float(segment["duration"]))
    if "start" in segment and "end" in segment:
        return max(0.0, float(segment["end"]) - float(segment["start"]))
    return 0.0


def counter_pairs(counter: Counter[str]) -> list[list[Any]]:
    return [[key, value] for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def downsample_scatter(points: list[list[float]], target: int) -> list[list[float]]:
    if len(points) <= target:
        return [[round(value, 4) for value in row] for row in points]
    stride = len(points) / target
    return [[round(value, 4) for value in points[int(index * stride)]] for index in range(target)]


def text_length_warnings(
    file_ids: list[str],
    name_by_file: dict[str, str],
    char_counts: list[int],
    min_chars: int,
    max_chars: int,
    limit: int,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for audio_id, char_count in zip(file_ids, char_counts, strict=True):
        if char_count == 0:
            reason = "empty transcript"
        elif char_count < min_chars:
            reason = "too short"
        elif char_count > max_chars:
            reason = "too long"
        else:
            continue
        warnings.append({"audio_file_id": audio_id, "name": name_by_file.get(audio_id, audio_id), "char_count": int(char_count), "reason": reason})
    warnings.sort(key=lambda item: (-item["char_count"] if item["reason"] == "too long" else item["char_count"]))
    return warnings[:limit]


def source_batch_id(records: list[dict[str, Any]]) -> str | None:
    batch_ids = {str(record["source_batch_id"]) for record in records if "source_batch_id" in record}
    if not batch_ids:
        return None
    assert len(batch_ids) == 1, f"mixed source batch ids: {sorted(batch_ids)}"
    return next(iter(batch_ids))


def source_batch_count(records: list[dict[str, Any]]) -> int:
    counts = {int(record["source_batch_count"]) for record in records if "source_batch_count" in record}
    assert len(counts) == 1, f"mixed source batch counts: {sorted(counts)}"
    return next(iter(counts))
