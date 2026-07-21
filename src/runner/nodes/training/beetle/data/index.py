import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from shared.db.audio.segment_references_crud import (
    SegmentCursor,
    SegmentReference,
    count_segment_references,
    list_segment_references_page,
)
from shared.db.connection import database_session

from ..config.data import DatabaseSelection
from .records import IndexedSegment, SegmentKey, WordBoundary
from .validation_types import ValidationCandidates, build_validation_candidates


class IndexCallbacks(Protocol):
    def check_cancel(self) -> None: ...

    def report_index_progress(self, scanned: int, total: int) -> None: ...


@dataclass(frozen=True)
class EligibilityReport:
    total_references: int
    stage1_eligible: int
    stage2_eligible: int
    stage3_eligible: int
    mid_sentence_eligible: int
    excluded_virtual: int
    excluded_external: int
    excluded_duration: int
    excluded_language: int
    excluded_text_or_voice: int

    def require(self, stage: int, sentence_probability: float) -> None:
        stage_count = {
            1: self.stage1_eligible,
            2: self.stage2_eligible,
            3: self.stage3_eligible,
        }[stage]
        if stage_count == 0:
            raise ValueError(
                f"Stage {stage} has no eligible segments: {self.describe()}"
            )
        if sentence_probability < 1 and self.mid_sentence_eligible == 0:
            raise ValueError(
                "mid-sentence sampling requested but no aligned segment maps "
                f"to phoneme word groups: {self.describe()}"
            )

    def describe(self) -> str:
        return (
            f"total={self.total_references}, stage1={self.stage1_eligible}, "
            f"stage2={self.stage2_eligible}, stage3={self.stage3_eligible}, "
            f"mid_sentence={self.mid_sentence_eligible}, virtual={self.excluded_virtual}, "
            f"external={self.excluded_external}, duration={self.excluded_duration}, "
            f"language={self.excluded_language}, "
            f"text_or_voice={self.excluded_text_or_voice}"
        )


@dataclass(frozen=True)
class StagePools:
    stage1: tuple[SegmentKey, ...]
    stage2: tuple[SegmentKey, ...]
    stage3: tuple[SegmentKey, ...]
    mid_sentence: tuple[SegmentKey, ...]
    voice_groups: dict[str, tuple[SegmentKey, ...]]
    recording_groups: dict[UUID, tuple[SegmentKey, ...]]

    def for_stage(self, stage: int) -> tuple[SegmentKey, ...]:
        return {1: self.stage1, 2: self.stage2, 3: self.stage3}[stage]


@dataclass(frozen=True)
class DatabaseSegmentIndex:
    dataset_id: UUID
    records: dict[SegmentKey, IndexedSegment]
    by_audio: dict[UUID, tuple[IndexedSegment, ...]]
    pools: StagePools
    validation: ValidationCandidates
    report: EligibilityReport
    fingerprint: str

    @classmethod
    def build(
        cls,
        selection: DatabaseSelection,
        languages: tuple[str, ...],
        maximum_seconds: float,
        page_size: int,
        callbacks: IndexCallbacks,
    ) -> "DatabaseSegmentIndex":
        references: list[SegmentReference] = []
        cursor: SegmentCursor | None = None
        with database_session() as session:
            total = count_segment_references(session, selection.dataset_id)
            while True:
                callbacks.check_cancel()
                page = list_segment_references_page(
                    session,
                    selection.dataset_id,
                    cursor,
                    page_size,
                )
                if not page:
                    break
                references.extend(page)
                cursor = page[-1].cursor
                callbacks.report_index_progress(len(references), total)
        return cls.from_references(
            selection.dataset_id,
            selection.audio_file_ids,
            languages,
            maximum_seconds,
            references,
        )

    @classmethod
    def from_references(
        cls,
        dataset_id: UUID,
        selected_audio_ids: tuple[UUID, ...],
        languages: tuple[str, ...],
        maximum_seconds: float,
        references: list[SegmentReference],
    ) -> "DatabaseSegmentIndex":
        selected = frozenset(selected_audio_ids)
        filtered = [
            reference
            for reference in references
            if not selected or reference.audio_file_id in selected
        ]
        validation = build_validation_candidates(filtered, languages)
        records: dict[SegmentKey, IndexedSegment] = {}
        by_audio_lists: dict[UUID, list[IndexedSegment]] = defaultdict(list)
        stage1: list[SegmentKey] = []
        stage2: list[SegmentKey] = []
        mid_sentence: list[SegmentKey] = []
        voice_lists: dict[str, list[SegmentKey]] = defaultdict(list)
        recording_lists: dict[UUID, list[SegmentKey]] = defaultdict(list)
        excluded_virtual = excluded_external = excluded_duration = 0
        excluded_language = 0
        excluded_text_or_voice = 0
        configured_languages = frozenset(languages)
        for reference in filtered:
            if reference.audio_virtual:
                excluded_virtual += 1
                continue
            if reference.audio_storage_kind != "packed":
                excluded_external += 1
                continue
            item = _indexed_segment(reference, reference.language)
            if item.duration > maximum_seconds:
                excluded_duration += 1
                continue
            if item.key in records:
                raise ValueError(f"duplicate segment key: {item.key}")
            records[item.key] = item
            by_audio_lists[item.key.audio_file_id].append(item)
            stage1.append(item.key)
            if (
                reference.language is None
                or reference.language not in configured_languages
            ):
                excluded_language += 1
                continue
            if item.has_text and item.has_phonemes and item.speaker_id is not None:
                stage2.append(item.key)
                voice_lists[item.speaker_id].append(item.key)
                recording_lists[item.key.audio_file_id].append(item.key)
                if item.mid_sentence_eligible:
                    mid_sentence.append(item.key)
            else:
                excluded_text_or_voice += 1
        by_audio = {
            audio_id: tuple(sorted(items, key=lambda item: item.key.segment_index))
            for audio_id, items in by_audio_lists.items()
        }
        pools = StagePools(
            stage1=tuple(sorted(stage1)),
            stage2=tuple(sorted(stage2)),
            stage3=tuple(sorted(stage2)),
            mid_sentence=tuple(sorted(mid_sentence)),
            voice_groups={key: tuple(sorted(value)) for key, value in voice_lists.items()},
            recording_groups={key: tuple(sorted(value)) for key, value in recording_lists.items()},
        )
        report = EligibilityReport(
            total_references=len(filtered),
            stage1_eligible=len(stage1),
            stage2_eligible=len(stage2),
            stage3_eligible=len(stage2),
            mid_sentence_eligible=len(mid_sentence),
            excluded_virtual=excluded_virtual,
            excluded_external=excluded_external,
            excluded_duration=excluded_duration,
            excluded_language=excluded_language,
            excluded_text_or_voice=excluded_text_or_voice,
        )
        return cls(
            dataset_id,
            records,
            by_audio,
            pools,
            validation,
            report,
            _fingerprint(dataset_id, records, validation),
        )


def _indexed_segment(
    reference: SegmentReference,
    language: str | None,
) -> IndexedSegment:
    raw = reference.segment
    key = SegmentKey(reference.audio_file_id, reference.segment_index, str(raw["id"]))
    start = float(raw["start"])
    end = float(raw["end"])
    text = str(raw["text"])
    phonemes = str(raw["phon"])
    speaker_id = str(raw["speaker_id"]) if "speaker_id" in raw and raw["speaker_id"] else None
    alignment = raw["alignment"] if "alignment" in raw and raw["alignment"] else []
    words = tuple(
        WordBoundary(str(item["word"]), float(item["start"]), float(item["end"]))
        for item in alignment
    )
    estimated = max(1, round(reference.audio_byte_length * (end - start) / reference.audio_duration))
    return IndexedSegment(
        key=key,
        audio_name=reference.audio_name,
        start=start,
        end=end,
        audio_duration=reference.audio_duration,
        sample_rate=int(reference.annotations.metadata["sample_rate"]),
        estimated_bytes=estimated,
        language=language,
        speaker_id=speaker_id,
        has_text=bool(text.strip()),
        has_phonemes=bool(phonemes.strip()),
        phoneme_word_count=len(phonemes.split()),
        words=words,
        style_prompt=reference.style_prompt,
        voice_prompt=reference.voice_prompt,
    )


def _fingerprint(
    dataset_id: UUID,
    records: dict[SegmentKey, IndexedSegment],
    validation: ValidationCandidates,
) -> str:
    rows = []
    for key in sorted(records):
        item = records[key]
        rows.append(
            (
                str(key.audio_file_id), key.segment_index, key.segment_id,
                item.start, item.end, item.sample_rate, item.language,
                item.speaker_id, item.has_text,
                item.has_phonemes, item.phoneme_word_count,
                tuple((word.word, word.start, word.end) for word in item.words),
                item.style_prompt, item.voice_prompt,
            )
        )
    validation_rows = (
        tuple(str(audio_id) for audio_id in validation.stage1),
        tuple(
            (
                voice_id,
                tuple(str(audio_id) for audio_id in audio_ids),
            )
            for voice_id, audio_ids in validation.conditional_by_voice.items()
        ),
    )
    payload = json.dumps(
        (str(dataset_id), rows, validation_rows),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
