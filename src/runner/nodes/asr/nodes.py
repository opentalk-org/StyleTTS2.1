from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.asr.batch import TemporaryAudioBatch
from runner.nodes.asr.canary import load_canary_model, transcribe_wavs_to_segments as canary_transcribe_wavs
from runner.nodes.asr.parakeet import (
    load_parakeet_model,
    transcribe_wavs_to_aligned_segments as parakeet_transcribe_aligned_wavs,
    transcribe_wavs_to_segments as parakeet_transcribe_wavs,
)
from runner.nodes.asr.transcript import audio_with_transcript_segments
from runner.nodes.asr.whisper import load_whisper_model, transcribe_wavs_to_segments as whisper_transcribe_wavs
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AudioPort, CheckpointRefPort
from runner.nodes.models import Audio, CheckpointRef, typed_checkpoint


TranscriptSpan = tuple[float, float, str, float | None]


class TranscribeSettings(StrictSettings):
    language: str = "auto"
    batch_size: int = Field(default=16, ge=1, le=128)


class WhisperTranscribeSettings(StrictSettings):
    language: str = Field(default="auto", title="Language")


class ParakeetTranscribeSettings(StrictSettings):
    batch_size: int = Field(default=16, ge=1, le=128)
    output_alignment: bool = Field(
        default=False,
        title="Output word alignment",
        description="Populate each segment's per-word alignment from Parakeet's word timestamps.",
    )


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
    INPUTS = {"checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST), "audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
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
        audios = [inputs["audio"] for inputs in batch]
        context.check_cancel()
        transcribed = await asyncio.to_thread(self._transcribe_batch, audios)
        await context.report_progress(
            self.id,
            len(audios),
            len(audios),
            f"{self.MODEL_NAME} transcribed {len(audios)}/{len(audios)}",
        )
        return [{"audio": audio} for audio in transcribed]

    async def _ensure_model(self, checkpoint: CheckpointRef) -> None:
        if self._model is not None and self._loaded_checkpoint_id == checkpoint.checkpoint_id:
            return
        self._model = await asyncio.to_thread(self._load_model_logged, checkpoint.path)
        self._loaded_checkpoint_id = checkpoint.checkpoint_id

    def _load_model_logged(self, checkpoint_dir: Path) -> Any:
        self.logger.info("loading %s model from checkpoint", self.MODEL_NAME)
        return self._load_model(checkpoint_dir)

    def _load_model(self, checkpoint_dir: Path) -> Any:
        raise NotImplementedError

    def _transcribe_batch(self, audios: list[Audio]) -> list[Audio]:
        durations = [audio.duration for audio in audios]
        with TemporaryAudioBatch(audios) as paths:
            span_batches = self._transcribe_paths(paths, durations)
        assert len(span_batches) == len(audios), f"{self.MODEL_NAME} batch output mismatch"
        return [
            audio_with_transcript_segments(
                self.MODEL_NAME,
                audio,
                spans,
                self._transcript_language(),
            )
            for audio, spans in zip(audios, span_batches, strict=True)
        ]

    def _transcribe_paths(
        self,
        paths: list[Path],
        durations_sec: list[float],
    ) -> list[list[TranscriptSpan]]:
        raise NotImplementedError

    def _transcript_language(self) -> str:
        return str(getattr(self.settings, "language", "auto"))


class WhisperTranscribeNode(TranscribeNode):
    NODE_TYPE = "WhisperTranscribe"
    DESCRIPTION = "Transcribe speech to timestamped text segments using an OpenAI Whisper checkpoint. Takes a checkpoint and audio, and outputs the same audio annotated with transcript segments. Good general-purpose multilingual ASR; set the language or leave it on auto-detect."
    MODEL_NAME = "whisper"
    SETTINGS = WhisperTranscribeSettings

    def _load_model(self, checkpoint_dir: Path) -> Any:
        return load_whisper_model(checkpoint_dir)

    def _transcribe_paths(self, paths: list[Path], durations_sec: list[float]) -> list[list[TranscriptSpan]]:
        return whisper_transcribe_wavs(self._model, paths, durations_sec, self.settings.language)


class ParakeetTranscribeNode(TranscribeNode):
    NODE_TYPE = "ParakeetTranscribe"
    DESCRIPTION = "Transcribe audio with an NVIDIA Parakeet checkpoint, producing timestamped transcript segments on the audio. Fast, accurate English ASR; optionally emit per-word alignment timestamps for downstream segmentation and forced-alignment work."
    MODEL_NAME = "parakeet"
    SETTINGS = ParakeetTranscribeSettings

    def _load_model(self, checkpoint_dir: Path) -> Any:
        return load_parakeet_model(checkpoint_dir)

    def _transcribe_paths(self, paths: list[Path], durations_sec: list[float]) -> list[list[TranscriptSpan]]:
        return parakeet_transcribe_wavs(
            self._model,
            paths,
            durations_sec,
            batch_size=self.settings.batch_size,
        )

    def _transcribe_batch(self, audios: list[Audio]) -> list[Audio]:
        if not self.settings.output_alignment:
            return super()._transcribe_batch(audios)
        durations = [audio.duration for audio in audios]
        with TemporaryAudioBatch(audios) as paths:
            aligned_batches = parakeet_transcribe_aligned_wavs(
                self._model,
                paths,
                durations,
                batch_size=self.settings.batch_size,
            )
        assert len(aligned_batches) == len(audios), "parakeet batch output mismatch"
        outputs = []
        for audio, aligned in zip(audios, aligned_batches, strict=True):
            spans = [
                (start, end, text, confidence)
                for start, end, text, confidence, _words in aligned
            ]
            alignments = [words for _start, _end, _text, _confidence, words in aligned]
            outputs.append(
                audio_with_transcript_segments(
                    self.MODEL_NAME,
                    audio,
                    spans,
                    self._transcript_language(),
                    alignments=alignments,
                )
            )
        return outputs


class CanaryTranscribeNode(TranscribeNode):
    NODE_TYPE = "CanaryTranscribe"
    DESCRIPTION = "Transcribe or translate speech with an NVIDIA Canary checkpoint, writing timestamped segments onto the audio. Supports many European languages and speech-to-text translation by choosing distinct source and target languages, with optional punctuation and capitalization."
    MODEL_NAME = "canary"
    SETTINGS = CanaryTranscribeSettings

    def _load_model(self, checkpoint_dir: Path) -> Any:
        return load_canary_model(checkpoint_dir)

    def _transcribe_paths(self, paths: list[Path], durations_sec: list[float]) -> list[list[TranscriptSpan]]:
        prompt = self._prompt_settings()
        return canary_transcribe_wavs(self._model, paths, durations_sec, **prompt)

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
