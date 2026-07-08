from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode, Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.asr.audio import write_temp_wav
from runner.nodes.asr.canary import load_canary_model, transcribe_wavs_to_segments as canary_transcribe_wavs
from runner.nodes.asr.parakeet import load_parakeet_model, transcribe_wavs_to_segments as parakeet_transcribe_wavs
from runner.nodes.asr.whisper import load_whisper_model, transcribe_wav_to_segments
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AUDIO, CHECKPOINT_REF
from runner.nodes.models import Audio, AudioSegment, CheckpointRef, stable_id, typed_checkpoint
from shared.log_streams import route_output_to_logger


class TranscribeSettings(StrictSettings):
    language: str = "auto"
    batch_size: int = Field(default=16, ge=1, le=128)


class WhisperTranscribeSettings(StrictSettings):
    language: str = Field(default="auto", title="Language")


class ParakeetTranscribeSettings(StrictSettings):
    batch_size: int = Field(default=16, ge=1, le=128)


class CanaryLanguage(str, Enum):
    AUTO = "auto"
    BG = "bg"
    HR = "hr"
    CS = "cs"
    DA = "da"
    NL = "nl"
    EN = "en"
    ET = "et"
    FI = "fi"
    FR = "fr"
    DE = "de"
    EL = "el"
    HU = "hu"
    IT = "it"
    LV = "lv"
    LT = "lt"
    MT = "mt"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    SK = "sk"
    SL = "sl"
    ES = "es"
    SV = "sv"
    RU = "ru"
    UK = "uk"


class CanaryTranscribeSettings(StrictSettings):
    language: CanaryLanguage = Field(default=CanaryLanguage.PL, title="Source language")
    target_language: CanaryLanguage = Field(default=CanaryLanguage.PL, title="Target language")
    punctuation_and_capitalization: bool = Field(default=True, title="Punctuation and capitalization")
    batch_size: int = Field(default=16, ge=1, le=128)


class TranscribeNode(Node):
    CATEGORY = "ASR"
    MODEL_NAME = "asr"
    SETTINGS = TranscribeSettings
    INPUTS = {"checkpoint": Port("checkpoint", CHECKPOINT_REF, join_mode=JoinMode.BROADCAST), "audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64, sort_by="duration")
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=True, exclusive_group="accelerator")

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._model: Any | None = None
        self._loaded_checkpoint_id: UUID | None = None

    async def teardown(self, context: Any) -> None:
        self._model = None
        self._loaded_checkpoint_id = None
        release_accelerator_memory()

    async def execute(self, batch, context):
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        await self._ensure_model(checkpoint)
        outputs = []
        for index, inputs in enumerate(batch, start=1):
            audio = inputs["audio"]
            await context.report_progress(self.id, index, len(batch), f"{self.MODEL_NAME} transcribed {index}/{len(batch)}")
            # route_output_to_logger wraps a synchronous call only (no await inside) so tqdm from
            # transcription reaches the node log without capturing other coroutines' output.
            with route_output_to_logger(self.logger):
                transcribed = self._transcribe_audio(audio)
            outputs.append({"audio": transcribed})
        return outputs

    async def _ensure_model(self, checkpoint: CheckpointRef) -> None:
        if self._model is not None and self._loaded_checkpoint_id == checkpoint.checkpoint_id:
            return
        self._model = await asyncio.to_thread(self._load_model_logged, checkpoint.path)
        self._loaded_checkpoint_id = checkpoint.checkpoint_id

    def _load_model_logged(self, checkpoint_dir: Path) -> Any:
        self.logger.info("loading %s model from checkpoint", self.MODEL_NAME)
        with route_output_to_logger(self.logger):
            return self._load_model(checkpoint_dir)

    def _load_model(self, checkpoint_dir: Path) -> Any:
        raise NotImplementedError

    def _transcribe_audio(self, audio: Audio) -> Audio:
        assert audio.data is not None, f"audio bytes are required for transcription: {audio.id}"
        path = write_temp_wav(audio.data)
        try:
            spans = self._transcribe_full_path(path, audio.duration)
        finally:
            path.unlink(missing_ok=True)
        return _audio_with_transcript_segments(self.MODEL_NAME, audio, spans, self._transcript_language())

    def _transcribe_full_path(self, path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
        raise NotImplementedError

    def _transcript_language(self) -> str:
        return str(getattr(self.settings, "language", "auto"))


class WhisperTranscribeNode(TranscribeNode):
    NODE_TYPE = "WhisperTranscribe"
    MODEL_NAME = "whisper"
    SETTINGS = WhisperTranscribeSettings

    def _load_model(self, checkpoint_dir: Path) -> Any:
        return load_whisper_model(checkpoint_dir)

    def _transcribe_full_path(self, path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
        return transcribe_wav_to_segments(self._model, path, duration_sec, self.settings.language)


class ParakeetTranscribeNode(TranscribeNode):
    NODE_TYPE = "ParakeetTranscribe"
    MODEL_NAME = "parakeet"
    SETTINGS = ParakeetTranscribeSettings

    def _load_model(self, checkpoint_dir: Path) -> Any:
        return load_parakeet_model(checkpoint_dir)

    def _transcribe_full_path(self, path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
        batches = parakeet_transcribe_wavs(self._model, [path], [duration_sec], batch_size=self.settings.batch_size)
        return batches[0] if batches else []


class CanaryTranscribeNode(TranscribeNode):
    NODE_TYPE = "CanaryTranscribe"
    MODEL_NAME = "canary"
    SETTINGS = CanaryTranscribeSettings

    def _load_model(self, checkpoint_dir: Path) -> Any:
        return load_canary_model(checkpoint_dir)

    def _transcribe_full_path(self, path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
        prompt = self._prompt_settings()
        batches = canary_transcribe_wavs(self._model, [path], [duration_sec], **prompt)
        return batches[0] if batches else []

    def _prompt_settings(self) -> dict[str, Any]:
        source_language = self.settings.language.value
        target_language = self.settings.target_language.value
        if source_language == CanaryLanguage.AUTO.value or target_language == CanaryLanguage.AUTO.value:
            raise ValueError("canary_language_auto_unsupported")
        return {
            "source_language": source_language,
            "target_language": target_language,
            "pnc": self.settings.punctuation_and_capitalization,
            "batch_size": self.settings.batch_size,
        }


def _audio_with_transcript_segments(
    model_name: str,
    audio: Audio,
    spans: list[tuple[float, float, str]],
    language: str,
) -> Audio:
    filtered = [
        (audio.start + start, audio.start + end, text)
        for start, end, raw_text in spans
        if (text := _clean_transcript_text(raw_text))
    ]
    text = _joined_text(filtered)
    transcript_id = stable_id("transcript", model_name, audio.audio_file_id, audio.id)
    speaker = _diarized_speaker(audio)
    segments = [
        AudioSegment(
            source_audio_id=audio.audio_file_id,
            name=f"{model_name}:{audio.name}",
            start=start,
            end=end,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            text=item_text,
            phon="",
            id=stable_id("segment", transcript_id, index),
            lineage_id=stable_id("segment_lineage", audio.lineage_id, transcript_id, index),
            segment_id=stable_id("transcript_segment", transcript_id, index),
            speaker=speaker,
            metadata={"transcript_id": transcript_id, "transcript_segment_index": index, "model": model_name, "type_": model_name},
        )
        for index, (start, end, item_text) in enumerate(filtered)
    ]
    return replace(
        audio,
        segments=segments,
        metadata={
            **audio.metadata,
            "language": language,
            "model": model_name,
            "type_": model_name,
            "transcript_id": transcript_id,
            "transcript_text": text,
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
        },
    )


def _clean_transcript_text(text: str) -> str:
    cleaned = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return cleaned if cleaned.strip(". \t\r\n") else ""


def _joined_text(spans: list[tuple[float, float, str]]) -> str:
    return " ".join(text.strip() for _start, _end, text in spans if text.strip()).strip()


def _diarized_speaker(audio: Audio) -> str | None:
    if "diarization" not in audio.metadata:
        return None
    speaker = audio.metadata.get("speaker")
    return str(speaker) if speaker else None
