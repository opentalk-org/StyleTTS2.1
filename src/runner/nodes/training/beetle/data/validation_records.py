from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from torch import Tensor

from shared.db.audio.segment_catalog import SegmentReference
from shared.db.audio.ranges.wav import WavClip

from .audio import ProcessedAudio
from .records import BeetleBatch


@dataclass(frozen=True)
class ValidationSegment:
    segment_id: str
    start: float
    end: float
    text: str
    phonemes: str
    speaker_id: str | None

@dataclass(frozen=True)
class StoredValidationAudio:
    audio_file_id: UUID
    duration: float
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    virtual: bool
    storage_kind: str
    segments: tuple[ValidationSegment, ...]
    clip: WavClip


@dataclass(frozen=True)
class PreparedValidationAudio:
    stored: StoredValidationAudio
    processed: ProcessedAudio


@dataclass(frozen=True)
class ValidationSource:
    items: tuple[PreparedValidationAudio, ...]


@dataclass(frozen=True)
class ValidationCandidates:
    audio_file_ids: tuple[UUID, ...]
    conditional_by_voice: dict[str, tuple[UUID, ...]]

    def voice_for(self, audio_file_id: UUID) -> str:
        matches = tuple(
            voice_id
            for voice_id, audio_ids in self.conditional_by_voice.items()
            if audio_file_id in audio_ids
        )
        if len(matches) != 1:
            raise KeyError(f"conditional validation audio is not indexed: {audio_file_id}")
        return matches[0]


@dataclass(frozen=True)
class ValidationRecording:
    audio_file_id: UUID
    batch: BeetleBatch
    waveform: Tensor


def build_validation_candidates(
    references: Iterable[SegmentReference],
    configured_languages: tuple[str, ...],
) -> ValidationCandidates:
    grouped: dict[UUID, list[SegmentReference]] = defaultdict(list)
    for reference in references:
        grouped[reference.audio_file_id].append(reference)
    languages = frozenset(configured_languages)
    conditional: dict[str, list[UUID]] = defaultdict(list)
    for audio_id in sorted(grouped):
        items = grouped[audio_id]
        if any(item.audio_virtual or item.audio_storage_kind != "packed" for item in items):
            continue
        voices = {_conditional_voice(item, languages) for item in items}
        if None not in voices and len(voices) == 1:
            conditional[next(iter(voices))].append(audio_id)
    return ValidationCandidates(
        tuple(
            sorted(
                audio_id
                for audio_ids in conditional.values()
                for audio_id in audio_ids
            )
        ),
        {
            voice_id: tuple(sorted(audio_ids))
            for voice_id, audio_ids in sorted(conditional.items())
        },
    )


def _conditional_voice(
    reference: SegmentReference,
    configured_languages: frozenset[str],
) -> str | None:
    segment = reference.segment
    voice = reference.annotations.speaker_id
    if (
        reference.language not in configured_languages
        or not str(segment["text"]).strip()
        or not str(segment["phon"]).strip()
        or voice is None
    ):
        return None
    return str(voice)
