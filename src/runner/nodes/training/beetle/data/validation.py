import random
from typing import Any
from uuid import UUID

from shared.audio_annotations import AudioAnnotations
from shared.db.audio import crud as audio_crud
from shared.db.audio.ranges.wav import WavTimeRange, slice_wav_ranges
from shared.db.connection import database_session

from ..config import BeetleConfig
from .audio import AudioPreprocessor
from .index import DatabaseSegmentIndex
from .records import SegmentKey
from .seeding import derive_seed
from .validation_collate import ValidationTokenizer, collate_validation_recording
from .validation_records import (
    PreparedValidationAudio,
    StoredValidationAudio,
    ValidationRecording,
    ValidationSegment,
    ValidationSource,
)


def select_validation_audio_ids(
    index: DatabaseSegmentIndex,
    sample_count: int,
    runtime_seed: int,
    configured_audio_file_ids: tuple[UUID, ...],
    require_distinct_voices: bool,
) -> tuple[UUID, ...]:
    candidates = list(index.validation.audio_file_ids)
    if len(candidates) < sample_count:
        raise ValueError(
            f"Validation requires {sample_count} recordings "
            f"but only {len(candidates)} are eligible"
        )
    if configured_audio_file_ids:
        if len(configured_audio_file_ids) != sample_count:
            raise ValueError(
                "configured validation audio count must equal validation sample_count"
            )
        if len(set(configured_audio_file_ids)) != sample_count:
            raise ValueError("configured validation audio IDs must be unique")
        missing = set(configured_audio_file_ids).difference(candidates)
        if missing:
            raise KeyError(f"configured validation audio IDs are not indexed: {missing}")
        selected = list(configured_audio_file_ids)
    else:
        rng = random.Random(derive_seed(runtime_seed, "validation-recordings"))
        rng.shuffle(candidates)
        selected = candidates[:sample_count]
    candidate_voices = {
        index.validation.voice_for(audio_id) for audio_id in candidates
    }
    if require_distinct_voices and (len(candidate_voices) < 2 or sample_count < 2):
        raise ValueError(
            "conditional validation requires at least two distinct voices"
        )
    selected_voices = {
        index.validation.voice_for(audio_id) for audio_id in selected
    }
    if require_distinct_voices and len(selected_voices) == 1:
        if configured_audio_file_ids:
            raise ValueError(
                "configured validation audio requires at least two distinct voices"
            )
        selected_voice = next(iter(selected_voices))
        replacement = next(
            audio_id
            for audio_id in candidates[sample_count:]
            if index.validation.voice_for(audio_id) != selected_voice
        )
        selected[-1] = replacement
    return tuple(selected)


class ValidationLoader:
    def __init__(self, config: BeetleConfig) -> None:
        self.config = config
        audio = config.audio
        self.preprocessor = AudioPreprocessor(
            audio.sample_rate,
            audio.n_fft,
            audio.win_length,
            audio.hop_length,
            audio.mel_channels,
            audio.f_min,
            audio.f_max,
        )

    def load_source(
        self,
        audio_file_ids: tuple[UUID, ...],
    ) -> ValidationSource:
        if not audio_file_ids or len(set(audio_file_ids)) != len(audio_file_ids):
            raise ValueError("validation audio_file_ids must be nonempty and unique")
        loaded = _load_stored_audio(audio_file_ids)
        missing = tuple(audio_id for audio_id in audio_file_ids if audio_id not in loaded)
        if missing:
            raise KeyError(f"validation audio files not found: {missing}")
        prepared = []
        for audio_id in audio_file_ids:
            stored = loaded[audio_id]
            if stored.audio_file_id != audio_id:
                raise ValueError(f"validation audio ID mismatch: {audio_id}")
            _validate_stored(stored, self.config)
            key = _validation_key(stored)
            try:
                processed = self.preprocessor.decode(stored.clip, key)
            except ValueError as error:
                raise ValueError(f"invalid validation audio {audio_id}: {error}") from error
            prepared.append(PreparedValidationAudio(stored, processed))
        return ValidationSource(tuple(prepared))

    def collate(
        self,
        source: ValidationSource,
        phoneme_tokenizer: ValidationTokenizer,
        text_tokenizer: ValidationTokenizer,
    ) -> tuple[ValidationRecording, ...]:
        return tuple(
            collate_validation_recording(
                self.config,
                item,
                phoneme_tokenizer,
                text_tokenizer,
            )
            for item in source.items
        )

    def load(
        self,
        audio_file_ids: tuple[UUID, ...],
        phoneme_tokenizer: ValidationTokenizer,
        text_tokenizer: ValidationTokenizer,
    ) -> tuple[ValidationRecording, ...]:
        source = self.load_source(audio_file_ids)
        return self.collate(source, phoneme_tokenizer, text_tokenizer)


def _load_stored_audio(
    audio_file_ids: tuple[UUID, ...],
) -> dict[UUID, StoredValidationAudio]:
    with database_session() as session:
        rows = audio_crud.get_audio_files_bulk(session, audio_file_ids)
        segments = audio_crud.list_audio_segments_bulk(session, audio_file_ids)
        for audio_id, row in rows.items():
            if row.virtual or row.storage_kind != "packed":
                raise ValueError(f"validation audio is not stored: {audio_id}")
        payloads = audio_crud.bulk_read_audio_files(session, audio_file_ids)
    return {
        audio_id: _stored_audio(
            rows[audio_id],
            segments[audio_id],
            payloads[audio_id],
        )
        for audio_id in audio_file_ids
    }


def _stored_audio(row: Any, segments: list[dict[str, Any]], payload: bytes) -> StoredValidationAudio:
    clips = slice_wav_ranges(payload, [WavTimeRange(0.0, float(row.duration))])
    if len(clips) != 1:
        raise ValueError(f"full validation WAV read failed: {row.id}")
    return StoredValidationAudio(
        audio_file_id=row.id,
        duration=float(row.duration),
        language=row.language,
        style_prompt=row.style_prompt,
        voice_prompt=row.voice_prompt,
        virtual=bool(row.virtual),
        storage_kind=str(row.storage_kind),
        segments=tuple(_segment(item) for item in segments),
        clip=clips[0],
    )


def _segment(value: dict[str, Any]) -> ValidationSegment:
    voice = AudioAnnotations.model_validate(value["annotations"]).speaker_id
    return ValidationSegment(
        str(value["id"]),
        float(value["start"]),
        float(value["end"]),
        str(value["text"]),
        str(value["phon"]),
        None if voice is None else str(voice),
    )


def _validate_stored(
    stored: StoredValidationAudio,
    config: BeetleConfig,
) -> None:
    audio_id = stored.audio_file_id
    if stored.virtual or stored.storage_kind != "packed":
        raise ValueError(f"validation audio is not readable: {audio_id}")
    if stored.duration <= 0:
        raise ValueError(f"validation audio duration is invalid: {audio_id}")
    previous_start = 0.0
    for segment in stored.segments:
        if segment.start < previous_start or segment.end > stored.duration:
            raise ValueError(f"validation segment order is invalid: {audio_id}")
        previous_start = segment.start
    if stored.language not in config.architecture.language.values:
        raise ValueError(f"validation language is incomplete: {audio_id}")
    if not stored.segments:
        raise ValueError(f"validation transcript is missing: {audio_id}")
    if any(
        not segment.text.strip()
        or not segment.phonemes.strip()
        or segment.speaker_id is None
        for segment in stored.segments
    ):
        raise ValueError(f"validation conditioning is incomplete: {audio_id}")
    voices = {segment.speaker_id for segment in stored.segments}
    if len(voices) != 1:
        raise ValueError(f"validation recording has mixed voices: {audio_id}")


def _validation_key(stored: StoredValidationAudio) -> SegmentKey:
    segment_id = (
        stored.segments[0].segment_id
        if stored.segments
        else f"validation-{stored.audio_file_id}"
    )
    return SegmentKey(stored.audio_file_id, 0, segment_id)
