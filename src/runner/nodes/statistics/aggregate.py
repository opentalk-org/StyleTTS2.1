from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.datatypes import JsonPort
from runner.nodes.statistics.aggregate_helpers import (
    break_histograms,
    char_bigram_matrix,
    char_trigram_extremes,
    char_unigram_counts,
    counter_pairs,
    downsample_scatter,
    finite_field_values,
    flatten_segments,
    float_value,
    histogram_counts,
    inter_word_silences,
    list_value,
    pooled_histogram,
    record_value,
    segment_audio_id,
    segment_duration,
    segment_phon,
    segment_speaker,
    source_batch_count,
    source_batch_id,
    string_value,
    text_length_warnings,
)


class AggregateStatisticsSettings(StrictSettings):
    histogram_bins: int = Field(default=50, ge=10, le=200)
    silence_threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)
    text_min_chars: int = Field(default=5, ge=0, le=10_000)
    text_max_chars: int = Field(default=500, ge=1, le=100_000)
    text_warnings_limit: int = Field(default=50, ge=1, le=1000)
    rate_scatter_points: int = Field(default=800, ge=100, le=5000)
    inter_word_silence_max_seconds: float = Field(default=1.0, ge=0.1, le=10.0)


class AggregateDatasetStatisticsNode(Node):
    NODE_TYPE = "AggregateDatasetStatistics"
    DESCRIPTION = "Roll up per-file audio feature records (and their speech segments) into a single dataset statistics summary. Produces counts, durations, speaker and voice breakdowns, histograms of loudness, silence and clipping, character and phoneme n-gram distributions, speaking-rate scatter data, and text-length warnings. Wire the feature records from the audio analysis node here to build the payload shown on the statistics dashboard."
    CATEGORY = "Audio"
    SETTINGS = AggregateStatisticsSettings
    INPUTS = {
        "feature_records": JsonPort(mode=PortMode.LIST),
    }
    OUTPUTS = {"statistics": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._pending_features: dict[str, list[dict[str, Any]]] = {}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            features = [record_value(item) for item in list_value(inputs["feature_records"])]
            ready = self._ready_feature_records(features)
            if ready is None:
                continue
            segments = flatten_segments(ready)
            outputs.append({"statistics": aggregate_dataset_statistics(ready, segments, self.settings)})
        return outputs

    def _ready_feature_records(self, features: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if not features:
            return None
        batch_id = source_batch_id(features)
        if batch_id is None:
            return features
        expected = source_batch_count(features)
        pending = self._pending_features.setdefault(batch_id, [])
        pending.extend(features)
        if len(pending) < expected:
            return None
        ready = list(pending)
        del self._pending_features[batch_id]
        return ready


def aggregate_dataset_statistics(features: list[Any], segments: list[Any], settings: AggregateStatisticsSettings) -> dict[str, Any]:
    feature_records = [record_value(item) for item in features]
    segment_records = [record_value(item) for item in segments]
    file_ids = [string_value(item, "audio_file_id") for item in feature_records]
    name_by_file = {string_value(item, "audio_file_id"): string_value(item, "name") for item in feature_records}
    file_ids.extend(sorted({audio_id for audio_id in (segment_audio_id(item) for item in segment_records) if audio_id not in file_ids}))
    text_by_file: dict[str, list[str]] = {audio_id: [] for audio_id in file_ids}
    phon_by_file: dict[str, list[str]] = {audio_id: [] for audio_id in file_ids}
    speaker_seconds: Counter[str] = Counter()
    speaker_chars: Counter[str] = Counter()
    speaker_phonemes: Counter[str] = Counter()
    speaker_samples: Counter[str] = Counter()
    voice_seconds: Counter[str] = Counter()
    voice_samples: Counter[str] = Counter()
    words_per_second: list[list[float]] = []
    chars_per_second: list[list[float]] = []
    ipa_words_per_second: list[list[float]] = []
    phonemes_per_second: list[list[float]] = []
    inter_word_silence_values: list[float] = []
    corpus_text: list[str] = []
    corpus_phon: list[str] = []
    for segment in segment_records:
        audio_id = segment_audio_id(segment)
        text = string_value(segment, "text")
        phon = segment_phon(segment)
        speaker = segment_speaker(segment)
        voice_id = str(segment.get("voice_id") or "")
        duration = segment_duration(segment)
        text_by_file.setdefault(audio_id, []).append(text)
        phon_by_file.setdefault(audio_id, []).append(phon)
        corpus_text.append(text)
        corpus_phon.append(phon)
        inter_word_silence_values.extend(inter_word_silences(segment, settings.inter_word_silence_max_seconds))
        speaker_seconds[speaker] += duration
        speaker_chars[speaker] += len(text)
        speaker_phonemes[speaker] += len(phon)
        speaker_samples[speaker] += 1
        if voice_id:
            voice_seconds[voice_id] += duration
            voice_samples[voice_id] += 1
        if duration > 0.0:
            words = int(segment.get("word_count", len(text.split())))
            chars = len(text)
            # Each row is [duration, rate, total] so the frontend can plot rate against either
            # clip duration or the total word/char count from the same sampled points.
            words_per_second.append([duration, words / duration, float(words)])
            chars_per_second.append([duration, chars / duration, float(chars)])
            if phon:
                ipa_words = len(phon.split())
                phonemes = len(phon)
                ipa_words_per_second.append([duration, ipa_words / duration, float(ipa_words)])
                phonemes_per_second.append([duration, phonemes / duration, float(phonemes)])
    durations = [float_value(item, "duration") for item in feature_records]
    transcript_by_file = {audio_id: " ".join(text_by_file[audio_id]) for audio_id in file_ids}
    ipa_by_file = {audio_id: " ".join(phon_by_file[audio_id]) for audio_id in file_ids}
    char_counts = [sum(len(part) for part in text_by_file[audio_id]) for audio_id in file_ids]
    phoneme_counts = [sum(len(part) for part in phon_by_file[audio_id]) for audio_id in file_ids]
    transcript_sentence_markers = [text.count(". ") + int(text.endswith(".")) for text in transcript_by_file.values()]
    transcript_comma_markers = [text.count(", ") + int(text.endswith(",")) for text in transcript_by_file.values()]
    ipa_sentence_markers = [text.count(". ") + int(text.endswith(".")) for text in ipa_by_file.values()]
    ipa_comma_markers = [text.count(", ") + int(text.endswith(",")) for text in ipa_by_file.values()]
    total_duration = sum(durations)
    duplicate_collapsed = sum(int(item.get("duplicate_segments_collapsed", 0)) for item in feature_records)
    phonemes_available = any(part for parts in phon_by_file.values() for part in parts)
    warnings = text_length_warnings(
        file_ids,
        name_by_file,
        char_counts,
        settings.text_min_chars,
        settings.text_max_chars,
        settings.text_warnings_limit,
    )
    computation_modes = {string_value(item, "computation_mode") for item in feature_records}
    sample_selections = {string_value(item, "sample_selection") for item in feature_records}
    acoustic_availability = {bool(item["acoustic_metrics_available"]) for item in feature_records}
    assert len(computation_modes) == 1, f"mixed statistics computation modes: {sorted(computation_modes)}"
    assert len(sample_selections) == 1, f"mixed statistics sample selections: {sorted(sample_selections)}"
    assert len(acoustic_availability) == 1, "mixed acoustic metric availability"
    computation_mode = next(iter(computation_modes))
    sample_selection = next(iter(sample_selections))
    acoustic_metrics_available = next(iter(acoustic_availability))
    assert acoustic_metrics_available == (computation_mode == "acoustic"), f"invalid acoustic availability for {computation_mode} mode"
    requested_count = feature_records[0]["sample_requested_count"]
    breaks = break_histograms(file_ids, segment_records, settings.histogram_bins)
    return {
        "version": 19,
        "computation_mode": computation_mode,
        "acoustic_metrics_available": acoustic_metrics_available,
        "sample_scope": {
            "selection": sample_selection,
            "requested_count": requested_count,
            "actual_count": len(feature_records),
        },
        "params": {
            "histogram_bins": settings.histogram_bins,
            "silence_threshold_db": settings.silence_threshold_db,
            "text_min_chars": settings.text_min_chars,
            "text_max_chars": settings.text_max_chars,
            "inter_word_silence_max_seconds": settings.inter_word_silence_max_seconds,
        },
        "audio_file_ids": file_ids,
        "file_count": len(feature_records),
        "segment_count": len(segment_records),
        "speaker_count": len(speaker_seconds),
        "total_duration_seconds": total_duration,
        "mean_duration_seconds": (total_duration / len(durations)) if durations else 0.0,
        "median_duration_seconds": float(median(durations)) if durations else 0.0,
        "total_char_count": sum(char_counts),
        "duplicate_segments_collapsed": duplicate_collapsed,
        "phonemes_available": phonemes_available,
        "text_length_warnings": warnings,
        **breaks,
        "duration_seconds_histogram": histogram_counts(durations, settings.histogram_bins),
        "char_count_per_file_histogram": histogram_counts([float(value) for value in char_counts], settings.histogram_bins),
        "phoneme_count_per_file_histogram": histogram_counts([float(value) for value in phoneme_counts], settings.histogram_bins),
        "transcript_sentence_marker_count_per_file_histogram": histogram_counts([float(value) for value in transcript_sentence_markers], settings.histogram_bins),
        "transcript_comma_marker_count_per_file_histogram": histogram_counts([float(value) for value in transcript_comma_markers], settings.histogram_bins),
        "ipa_sentence_marker_count_per_file_histogram": histogram_counts([float(value) for value in ipa_sentence_markers], settings.histogram_bins),
        "ipa_comma_marker_count_per_file_histogram": histogram_counts([float(value) for value in ipa_comma_markers], settings.histogram_bins),
        "char_unigram_counts": char_unigram_counts(" ".join(corpus_text)),
        "phoneme_unigram_counts": char_unigram_counts(" ".join(corpus_phon)),
        "char_bigram_matrix": char_bigram_matrix(" ".join(corpus_text)),
        "phoneme_bigram_matrix": char_bigram_matrix(" ".join(corpus_phon)),
        "char_trigram_top10": char_trigram_extremes(" ".join(corpus_text))[0],
        "char_trigram_bottom10": char_trigram_extremes(" ".join(corpus_text))[1],
        "phoneme_trigram_top10": char_trigram_extremes(" ".join(corpus_phon))[0],
        "phoneme_trigram_bottom10": char_trigram_extremes(" ".join(corpus_phon))[1],
        "speaker_char_count": counter_pairs(speaker_chars),
        "speaker_phoneme_count": counter_pairs(speaker_phonemes),
        "speaker_sample_count": counter_pairs(speaker_samples),
        "voice_duration_seconds_histogram": histogram_counts(list(voice_seconds.values()), settings.histogram_bins),
        "voice_sample_count_histogram": histogram_counts([float(value) for value in voice_samples.values()], settings.histogram_bins),
        "words_per_second_scatter": downsample_scatter(words_per_second, settings.rate_scatter_points),
        "chars_per_second_scatter": downsample_scatter(chars_per_second, settings.rate_scatter_points),
        "ipa_words_per_second_scatter": downsample_scatter(ipa_words_per_second, settings.rate_scatter_points),
        "phonemes_per_second_scatter": downsample_scatter(phonemes_per_second, settings.rate_scatter_points),
        "inter_word_silence_seconds_histogram": histogram_counts(inter_word_silence_values, settings.histogram_bins, (0.0, settings.inter_word_silence_max_seconds)),
        "rms_db_histogram": pooled_histogram(feature_records, "rms_db", settings.histogram_bins, (-100.0, 0.0), clip=True),
        "frame_value_min_histogram": pooled_histogram(feature_records, "frame_value_min", settings.histogram_bins, (-1.0, 1.0)),
        "frame_value_max_histogram": pooled_histogram(feature_records, "frame_value_max", settings.histogram_bins, (-1.0, 1.0)),
        "frame_value_mean_histogram": pooled_histogram(feature_records, "frame_value_mean", settings.histogram_bins, (-1.0, 1.0)),
        "mean_rms_nonsilent_db_per_file_histogram": histogram_counts(finite_field_values(feature_records, "mean_rms_db_nonsilent"), settings.histogram_bins, (-80.0, 0.0)),
        "sample_rms_nonsilent_db_per_file_histogram": histogram_counts(finite_field_values(feature_records, "rms_db_nonsilent_samples"), settings.histogram_bins, (-80.0, 0.0)),
        "clipped_audio_file_count": sum(1 for item in feature_records if bool(item["has_clip"])),
        "clipped_sample_count_top": sum(int(item["clip_top"]) for item in feature_records),
        "clipped_sample_count_bottom": sum(int(item["clip_bottom"]) for item in feature_records),
        "silence_ratio_histogram": histogram_counts(finite_field_values(feature_records, "silence_ratio"), settings.histogram_bins, (0.0, 1.0)),
        "silence_rms_db_histogram": pooled_histogram(feature_records, "silence_rms_db", settings.histogram_bins, (-100.0, settings.silence_threshold_db), clip=True),
        "per_file_text": [{"audio_file_id": audio_id, "text": transcript_by_file[audio_id], "phon": ipa_by_file[audio_id]} for audio_id in file_ids],
    }
