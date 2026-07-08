from __future__ import annotations

import random
import uuid
from typing import Any, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode, Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.asr.audio import extract_wav_range, wav_info
from runner.nodes.audio_processing.nodes import (
    SortformerSettings,
    diarize_audio_batch,
    load_sortformer_model,
)
from runner.nodes.datatypes import AUDIO, CHECKPOINT_REF
from runner.nodes.models import Audio, AudioSegment, stable_id, typed_checkpoint
from shared.db import database_session
from shared.db.voices import crud as voice_crud
from shared.db.voices.schemas import VoiceCreate
from shared.log_streams import route_output_to_logger

_PUNCTUATION = ".,!?;:…—"


class DiarizeSplitSpeakersSettings(StrictSettings):
    """Diarize an already-transcribed audio, assign a random voice per speaker,
    and split the transcript into single-speaker clips whose durations follow a
    normal distribution over ``[min_segment_sec, max_segment_sec]``. Splits only
    happen at transcript segment boundaries that end on punctuation, so clips are
    never cut mid-word."""

    min_segment_sec: float = Field(default=1.0, ge=0.1, le=60.0)
    max_segment_sec: float = Field(default=14.0, ge=1.0, le=120.0)
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=4, ge=1, le=64)
    create_voices: bool = True
    voice_prefix: str = "spk"
    punctuation: str = _PUNCTUATION
    transcript_type: str = "parakeet"


class DiarizeSplitSpeakersNode(Node):
    NODE_TYPE = "DiarizeSplitSpeakers"
    CATEGORY = "Audio"
    SETTINGS = DiarizeSplitSpeakersSettings
    INPUTS = {
        "audio": Port("audio", AUDIO),
        "checkpoint": Port("checkpoint", CHECKPOINT_REF, join_mode=JoinMode.BROADCAST),
    }
    OUTPUTS = {"audio": Port("audio", AUDIO, mode=PortMode.STREAM)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=2, max_size=8)
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=True, exclusive_group="accelerator"
    )

    def __init__(self, node_id: str | None = None, **params: Any) -> None:
        super().__init__(node_id=node_id, **params)
        self._model: Any | None = None
        self._loaded_checkpoint_id: uuid.UUID | None = None

    async def teardown(self, context: Any) -> None:
        self._model = None
        self._loaded_checkpoint_id = None
        release_accelerator_memory()

    def _sortformer_settings(self) -> SortformerSettings:
        return SortformerSettings(
            batch_size=self.settings.batch_size,
            sample_rate=self.settings.sample_rate,
            min_segment_sec=0.25,
            device=self.settings.device,
        )

    async def execute(self, batch, context):
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        sort_settings = self._sortformer_settings()
        if self._model is None or self._loaded_checkpoint_id != checkpoint.checkpoint_id:
            with route_output_to_logger(self.logger):
                self._model = load_sortformer_model(checkpoint.path, sort_settings)
            self._loaded_checkpoint_id = checkpoint.checkpoint_id

        audios = [inputs["audio"] for inputs in batch]
        with route_output_to_logger(self.logger):
            diarized = diarize_audio_batch(self._model, audios, sort_settings)

        outputs: list[dict[str, list[Audio]]] = []
        with database_session() as session:
            for audio, diar in zip(audios, diarized, strict=True):
                clips = self._split_audio(session, audio, diar)
                outputs.append({"audio": clips})
        return outputs

    def _split_audio(self, session, audio: Audio, diar) -> list[Audio]:
        assert audio.data is not None, f"audio bytes are required: {audio.id}"
        # Transcript segments carry absolute (source) timestamps; convert to the
        # clip-local timeline that the diarization output uses.
        local_segments = sorted(
            (
                _LocalSegment(
                    start=max(0.0, segment.start - audio.start),
                    end=max(0.0, segment.end - audio.start),
                    text=segment.text,
                    segment=segment,
                )
                for segment in audio.segments
                if segment.text.strip()
            ),
            key=lambda item: item.start,
        )
        if not local_segments:
            return []

        for item in local_segments:
            item.speaker = _speaker_for_span(item.start, item.end, diar)
        voice_map = self._voice_map(session, audio, {item.speaker for item in local_segments})

        groups = _group_by_speaker_punctuation(local_segments, self.settings)
        info = wav_info(audio.data)
        clips: list[Audio] = []
        for index, group in enumerate(groups):
            clip = self._build_clip(audio, group, info, index, voice_map)
            if clip is not None:
                clips.append(clip)
        return clips

    def _voice_map(self, session, audio: Audio, speakers) -> dict[str, tuple[str, uuid.UUID | None]]:
        mapping: dict[str, tuple[str, uuid.UUID | None]] = {}
        for speaker in sorted(speakers):
            if not self.settings.create_voices:
                mapping[speaker] = (speaker, None)
                continue
            token = stable_id("voice", audio.audio_file_id, audio.id, speaker)[:12]
            name = f"{self.settings.voice_prefix}_{token}"
            voice = _get_or_create_voice(session, name)
            mapping[speaker] = (name, voice.id)
        return mapping

    def _build_clip(self, audio: Audio, group, info, index, voice_map) -> Audio | None:
        local_start = min(item.start for item in group)
        local_end = max(item.end for item in group)
        duration = local_end - local_start
        if duration <= 0.0:
            return None
        data = extract_wav_range(audio.data, local_start, local_end, info)
        speaker = group[0].speaker
        display_speaker, voice_id = voice_map.get(speaker, (speaker, None))
        text = " ".join(item.text.strip() for item in group if item.text.strip()).strip()
        clip_id = stable_id("audio", audio.audio_file_id, audio.id, index, local_start, local_end)
        transcript_type = self.settings.transcript_type
        segment = AudioSegment(
            source_audio_id=audio.audio_file_id,
            name=f"{transcript_type}:{audio.name}",
            start=0.0,
            end=duration,
            sample_rate=int(info["sample_rate"]),
            channels=int(info["channels"]),
            text=text,
            phon="",
            id=stable_id("segment", clip_id, transcript_type),
            lineage_id=stable_id("segment_lineage", audio.lineage_id, clip_id, transcript_type),
            segment_id=stable_id("transcript_segment", clip_id, transcript_type),
            speaker=display_speaker,
            voice_id=voice_id,
            confidence=audio.confidence,
            metadata={"model": transcript_type, "type_": transcript_type},
        )
        metadata = {
            **audio.metadata,
            "speaker": display_speaker,
            "voice_id": str(voice_id) if voice_id is not None else None,
            "diarization": {"model": "sortformer", "speaker": speaker},
            "source_audio_id": str(audio.audio_file_id),
            "source_audio_node_id": audio.id,
            "split_local_start": audio.start + local_start,
            "split_local_end": audio.start + local_end,
            "transcript_text": text,
            "sample_rate": int(info["sample_rate"]),
            "channels": int(info["channels"]),
        }
        return Audio(
            audio_file_id=audio.audio_file_id,
            name=f"{audio.name}_spk_{index + 1:04d}",
            data=data,
            sample_rate=int(info["sample_rate"]),
            channels=int(info["channels"]),
            start=0.0,
            end=duration,
            confidence=audio.confidence,
            id=clip_id,
            lineage_id=stable_id("lineage", audio.lineage_id, clip_id),
            metadata=metadata,
            byte_length=len(data),
            virtual=audio.virtual,
            segments=[segment],
        )


class _LocalSegment:
    __slots__ = ("start", "end", "text", "segment", "speaker")

    def __init__(self, start: float, end: float, text: str, segment: AudioSegment) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.segment = segment
        self.speaker = "speaker_0"


def _speaker_for_span(start: float, end: float, diar) -> str:
    best_speaker = "speaker_0"
    best_overlap = 0.0
    for segment in diar:
        overlap = min(end, segment.end) - max(start, segment.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(segment.speaker)
    return best_speaker


def _ends_on_punctuation(text: str, punctuation: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in punctuation


def _group_by_speaker_punctuation(segments, settings: DiarizeSplitSpeakersSettings):
    low = float(settings.min_segment_sec)
    high = max(low, float(settings.max_segment_sec))
    mean = (low + high) / 2.0
    std = max((high - low) / 4.0, 0.5)
    rng = random.Random(_group_seed(segments))

    groups: list[list[_LocalSegment]] = []
    index = 0
    total = len(segments)
    while index < total:
        target = min(high, max(low, rng.gauss(mean, std)))
        current: list[_LocalSegment] = []
        group_speaker = segments[index].speaker
        group_start = segments[index].start
        cursor = index
        while cursor < total:
            segment = segments[cursor]
            if current and segment.speaker != group_speaker:
                break  # speaker change always closes the current run
            current.append(segment)
            cursor += 1
            duration = segment.end - group_start
            ends_punct = _ends_on_punctuation(segment.text, settings.punctuation)
            if duration >= target and ends_punct:
                break
            if duration >= high and ends_punct:
                break
        groups.append(current)
        index = cursor
    return _merge_short_tail_groups(groups, low)


def _merge_short_tail_groups(groups, min_seconds: float):
    merged: list[list[_LocalSegment]] = []
    for group in groups:
        duration = max(item.end for item in group) - min(item.start for item in group)
        if (
            merged
            and duration < min_seconds
            and merged[-1][0].speaker == group[0].speaker
        ):
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    return merged


def _group_seed(segments) -> int:
    if not segments:
        return 0
    raw = f"{segments[0].start:.3f}|{segments[-1].end:.3f}|{len(segments)}"
    return int(stable_id("seed", raw).split("_")[1], 16)


def _get_or_create_voice(session, name: str):
    for voice in voice_crud.list_voices(session):
        if voice.name == name:
            return voice
    return voice_crud.create_voice(session, VoiceCreate(name=name))
