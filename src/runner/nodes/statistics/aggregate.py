from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import JSON


class AggregateStatisticsSettings(StrictSettings):
    histogram_bins: int = Field(default=50, ge=10, le=200)
    silence_threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)


class AggregateDatasetStatisticsNode(Node):
    NODE_TYPE = "AggregateDatasetStatistics"
    CATEGORY = "Audio / Statistics"
    SETTINGS = AggregateStatisticsSettings
    INPUTS = {
        "feature_records": Port("feature_records", JSON, mode=PortMode.LIST),
        "segment_records": Port("segment_records", JSON, mode=PortMode.LIST, optional=True, default=[]),
    }
    OUTPUTS = {"statistics": Port("statistics", JSON)}

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._pending_features: dict[str, list[dict[str, Any]]] = {}
        self._pending_segments: dict[str, list[dict[str, Any]]] = {}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            features = [_record(item) for item in _list_value(inputs["feature_records"])]
            segments = [_record(item) for item in _flatten_segments(_list_value(inputs["segment_records"]))]
            ready = self._ready_feature_records(features, segments)
            if ready is None:
                continue
            ready_features, ready_segments = ready
            outputs.append({"statistics": aggregate_dataset_statistics(ready_features, ready_segments, self.settings)})
        return outputs

    def _ready_feature_records(
        self,
        features: list[dict[str, Any]],
        segments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        if not features:
            return features, segments
        batch_id = _source_batch_id(features)
        if batch_id is None:
            return features, segments
        expected = _source_batch_count(features)
        pending = self._pending_features.setdefault(batch_id, [])
        pending_segments = self._pending_segments.setdefault(batch_id, [])
        pending.extend(features)
        pending_segments.extend(segments)
        if len(pending) < expected:
            return None
        ready = list(pending)
        ready_segments = list(pending_segments)
        del self._pending_features[batch_id]
        del self._pending_segments[batch_id]
        return ready, ready_segments


def aggregate_dataset_statistics(features: list[Any], segments: list[Any], settings: AggregateStatisticsSettings) -> dict[str, Any]:
    feature_records = [_record(item) for item in features]
    segment_records = [_record(item) for item in _flatten_segments(segments)]
    file_ids = [_string_value(item, "audio_file_id") for item in feature_records]
    file_ids.extend(sorted({audio_id for audio_id in (_segment_audio_id(item) for item in segment_records) if audio_id not in file_ids}))
    text_by_file: dict[str, list[str]] = {audio_id: [] for audio_id in file_ids}
    phon_by_file: dict[str, list[str]] = {audio_id: [] for audio_id in file_ids}
    speaker_seconds: Counter[str] = Counter()
    speaker_chars: Counter[str] = Counter()
    speaker_phonemes: Counter[str] = Counter()
    corpus_text: list[str] = []
    corpus_phon: list[str] = []
    for segment in segment_records:
        audio_id = _segment_audio_id(segment)
        text = _string_value(segment, "text")
        phon = _segment_phon(segment)
        speaker = _segment_speaker(segment)
        duration = _segment_duration(segment)
        text_by_file.setdefault(audio_id, []).append(text)
        phon_by_file.setdefault(audio_id, []).append(phon)
        corpus_text.append(text)
        corpus_phon.append(phon)
        speaker_seconds[speaker] += duration
        speaker_chars[speaker] += len(text)
        speaker_phonemes[speaker] += len(phon)
    durations = [_float_value(item, "duration") for item in feature_records]
    char_counts = [sum(len(part) for part in text_by_file[audio_id]) for audio_id in file_ids]
    phoneme_counts = [sum(len(part) for part in phon_by_file[audio_id]) for audio_id in file_ids]
    return {
        "version": 8,
        "params": {"histogram_bins": settings.histogram_bins, "silence_threshold_db": settings.silence_threshold_db},
        "audio_file_ids": file_ids,
        "file_count": len(feature_records),
        "duration_seconds_histogram": histogram_counts(durations, settings.histogram_bins),
        "char_count_per_file_histogram": histogram_counts([float(value) for value in char_counts], settings.histogram_bins),
        "phoneme_count_per_file_histogram": histogram_counts([float(value) for value in phoneme_counts], settings.histogram_bins),
        "char_unigram_counts": char_unigram_counts(" ".join(corpus_text)),
        "phoneme_unigram_counts": char_unigram_counts(" ".join(corpus_phon)),
        "char_bigram_matrix": char_bigram_matrix(" ".join(corpus_text)),
        "phoneme_bigram_matrix": char_bigram_matrix(" ".join(corpus_phon)),
        "char_trigram_top10": char_trigram_extremes(" ".join(corpus_text))[0],
        "char_trigram_bottom10": char_trigram_extremes(" ".join(corpus_text))[1],
        "phoneme_trigram_top10": char_trigram_extremes(" ".join(corpus_phon))[0],
        "phoneme_trigram_bottom10": char_trigram_extremes(" ".join(corpus_phon))[1],
        "speaker_duration_seconds": _counter_pairs(speaker_seconds),
        "speaker_char_count": _counter_pairs(speaker_chars),
        "speaker_phoneme_count": _counter_pairs(speaker_phonemes),
        "rms_db_histogram": pooled_histogram(feature_records, "rms_db", settings.histogram_bins, (-80.0, 0.0)),
        "frame_value_min_histogram": pooled_histogram(feature_records, "frame_value_min", settings.histogram_bins, (-1.0, 1.0)),
        "frame_value_max_histogram": pooled_histogram(feature_records, "frame_value_max", settings.histogram_bins, (-1.0, 1.0)),
        "frame_value_mean_histogram": pooled_histogram(feature_records, "frame_value_mean", settings.histogram_bins, (-1.0, 1.0)),
        "mean_rms_nonsilent_db_per_file_histogram": histogram_counts(_finite_field_values(feature_records, "mean_rms_db_nonsilent"), settings.histogram_bins, (-80.0, 0.0)),
        "sample_rms_nonsilent_db_per_file_histogram": histogram_counts(_finite_field_values(feature_records, "rms_db_nonsilent_samples"), settings.histogram_bins, (-80.0, 0.0)),
        "clipped_audio_file_count": sum(1 for item in feature_records if bool(item["has_clip"])),
        "clipped_sample_count_top": sum(int(item["clip_top"]) for item in feature_records),
        "clipped_sample_count_bottom": sum(int(item["clip_bottom"]) for item in feature_records),
        "silence_ratio_histogram": histogram_counts(_finite_field_values(feature_records, "silence_ratio"), settings.histogram_bins, (0.0, 1.0)),
        "silence_rms_db_histogram": pooled_histogram(feature_records, "silence_rms_db", settings.histogram_bins, (-80.0, settings.silence_threshold_db)),
        "per_file_text": [{"audio_file_id": audio_id, "text": " ".join(text_by_file[audio_id]), "phon": " ".join(phon_by_file[audio_id])} for audio_id in file_ids],
    }


def histogram_counts(values: list[float], bins: int, range_: tuple[float, float] | None = None) -> dict[str, Any]:
    finite_values = [float(value) for value in values if isfinite(float(value))]
    lo, hi = _range_for_pair(range_) if range_ is not None else _range_for(finite_values)
    edges = [lo + ((hi - lo) * index / bins) for index in range(bins + 1)]
    counts = [0] * bins
    for value in finite_values:
        if value < lo or value > hi:
            continue
        slot = bins - 1 if value == hi else int((value - lo) / (hi - lo) * bins)
        counts[max(0, min(bins - 1, slot))] += 1
    return {"edges": edges, "counts": counts}


def pooled_histogram(records: list[dict[str, Any]], field: str, bins: int, fallback: tuple[float, float]) -> dict[str, Any]:
    values: list[float] = []
    for record in records:
        values.extend(float(value) for value in record[field] if isfinite(float(value)))
    value_range = _range_for(values) if values else fallback
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


def _flatten_segments(items: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in items:
        record = _record(item)
        if "segments" in record:
            flattened.extend(_segments_with_parent_audio_id(record))
            continue
        flattened.append(record)
    return flattened


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _record(item: Any) -> dict[str, Any]:
    assert isinstance(item, dict), f"statistics JSON record must be a dict, got {type(item).__name__}"
    return item


def _segments_with_parent_audio_id(record: dict[str, Any]) -> list[dict[str, Any]]:
    segments = record["segments"]
    assert isinstance(segments, list), "segments wrapper must contain a list"
    parent_audio_id = _parent_audio_id(record)
    out = []
    for item in segments:
        segment = _record(item)
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


def _range_for_pair(value_range: tuple[float, float]) -> tuple[float, float]:
    lo, hi = value_range
    if hi > lo:
        return lo, hi
    epsilon = 1e-9 if lo == 0 else abs(lo) * 1e-9
    return lo - epsilon, hi + epsilon


def _finite_field_values(records: list[dict[str, Any]], field: str) -> list[float]:
    return [float(record[field]) for record in records if record[field] is not None and isfinite(float(record[field]))]


def _string_value(record: dict[str, Any], field: str) -> str:
    value = record[field]
    if value is None:
        return ""
    return str(value)


def _float_value(record: dict[str, Any], field: str) -> float:
    return float(record[field])


def _segment_audio_id(segment: dict[str, Any]) -> str:
    for field in ("audio_file_id", "source_audio_id", "source_id"):
        if field in segment:
            return str(segment[field])
    raise KeyError("segment record missing audio_file_id/source_audio_id/source_id")


def _segment_phon(segment: dict[str, Any]) -> str:
    for field in ("phon", "phonemes"):
        if field in segment:
            return "" if segment[field] is None else str(segment[field])
    return ""


def _segment_speaker(segment: dict[str, Any]) -> str:
    if "speaker" in segment and str(segment["speaker"]).strip():
        return str(segment["speaker"]).strip()
    return "-"


def _segment_duration(segment: dict[str, Any]) -> float:
    if "duration" in segment:
        return max(0.0, float(segment["duration"]))
    if "start" in segment and "end" in segment:
        return max(0.0, float(segment["end"]) - float(segment["start"]))
    return 0.0


def _counter_pairs(counter: Counter[str]) -> list[list[Any]]:
    return [[key, value] for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def _source_batch_id(records: list[dict[str, Any]]) -> str | None:
    batch_ids = {str(record["source_batch_id"]) for record in records if "source_batch_id" in record}
    if not batch_ids:
        return None
    assert len(batch_ids) == 1, f"mixed source batch ids: {sorted(batch_ids)}"
    return next(iter(batch_ids))


def _source_batch_count(records: list[dict[str, Any]]) -> int:
    counts = {int(record["source_batch_count"]) for record in records if "source_batch_count" in record}
    assert len(counts) == 1, f"mixed source batch counts: {sorted(counts)}"
    return next(iter(counts))
