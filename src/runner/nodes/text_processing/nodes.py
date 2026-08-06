from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import chain
from math import ceil
from typing import Any, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, AudioSegment
from runner.nodes.text.runtime.phonemize import (
    DEFAULT_PUNCTUATION_MARKS,
    phonemize_texts,
    phonemizer_backend,
)

MAX_PHONEMIZE_SEGMENT_CHARACTERS = 2_048


class PhonemizeSettings(StrictSettings):
    punctuation_marks: str = Field(default=DEFAULT_PUNCTUATION_MARKS, min_length=1, max_length=512)
    processes: int = Field(default=2, ge=2, le=16)


class PhonemizeSegmentsSettings(PhonemizeSettings):
    mode: Literal["fill", "replace"] = "fill"


@dataclass(frozen=True, slots=True)
class PhonemizeJob:
    texts: list[str]
    language: str


class PhonemizeSegmentsNode(Node):
    NODE_TYPE = "PhonemizeSegments"
    DESCRIPTION = "Convert segment text into language-aware phonemes using the language stored on each incoming audio record: OpenJTalk for Japanese, G2PW for Mandarin, Korean pronunciation rules plus eSpeak for Korean, dedicated eSpeak Cantonese, and eSpeak IPA for other supported languages. Missing languages are rejected. In fill mode only segments without phonemes are processed; replace mode re-phonemizes everything."
    CATEGORY = "Text"
    SETTINGS = PhonemizeSegmentsSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=256,
        max_size=512,
        timeout_ms=25,
    )
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 2}, keep_loaded=True)
    QUEUE_MAX_SIZE = 1_024

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._executor: ProcessPoolExecutor | None = None

    async def setup(self, context: Any) -> None:
        self._executor = ProcessPoolExecutor(
            max_workers=self.settings.processes,
            mp_context=multiprocessing.get_context("spawn"),
        )

    async def teardown(self, context: Any) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    async def execute(self, batch, context):
        if self._executor is None:
            raise RuntimeError("PhonemizeSegments process pool is not loaded")
        loop = asyncio.get_running_loop()
        audios = []
        for input_index, inputs in enumerate(batch):
            audio: Audio = inputs["audio"]
            language = _audio_language(audio)
            work_segments = [segment for segment in audio.segments if _should_phonemize_segment(segment, self.settings)]
            if any(len(segment.text) > MAX_PHONEMIZE_SEGMENT_CHARACTERS for segment in work_segments):
                continue
            job = PhonemizeJob(
                texts=[segment.text for segment in work_segments],
                language=language,
            )
            audios.append((input_index, audio, language, job))
        if not audios:
            return []
        chunk_size = ceil(len(audios) / (self.settings.processes * 2))
        chunks = [
            [entry[3] for entry in audios[start:start + chunk_size]]
            for start in range(0, len(audios), chunk_size)
        ]
        futures = [
            loop.run_in_executor(
                self._executor,
                _phonemize_chunk_worker,
                chunk,
                self.settings.punctuation_marks,
            )
            for chunk in chunks
        ]
        phoneme_batches = list(chain.from_iterable(await asyncio.gather(*futures)))
        context.check_cancel()
        outputs = []
        for (input_index, audio, language, _job), phonemes in zip(audios, phoneme_batches, strict=True):
            segments = _replace_phonemes(audio.segments, self.settings, phonemes)
            metadata = {
                **audio.metadata,
                "phoneme_language": language,
                "phoneme_backend": phonemizer_backend(language),
                "punctuation_marks": self.settings.punctuation_marks,
                "phoneme_mode": self.settings.mode,
            }
            outputs.append({
                INPUT_INDEX_OUTPUT: input_index,
                "audio": replace(
                    audio,
                    segments=segments,
                    annotations=audio.annotations.model_copy(update={"metadata": metadata}),
                ),
            })
        return outputs


def _replace_phonemes(
    segments: list[AudioSegment],
    settings: PhonemizeSegmentsSettings,
    phonemes: list[str],
) -> list[AudioSegment]:
    phoneme_iter = iter(phonemes)
    return [
        replace(segment, phon=next(phoneme_iter))
        if _should_phonemize_segment(segment, settings)
        else segment
        for segment in segments
    ]


def _should_phonemize_segment(segment: AudioSegment, settings: PhonemizeSegmentsSettings) -> bool:
    return bool(segment.text.strip()) and (settings.mode == "replace" or not segment.phon.strip())


def _phonemize_chunk_worker(jobs: list[PhonemizeJob], punctuation_marks: str) -> list[list[str]]:
    return [
        phonemize_texts(
            job.texts,
            language=job.language,
            punctuation_marks=punctuation_marks,
        )
        for job in jobs
    ]


def _audio_language(audio: Audio) -> str:
    value = audio.language
    if value is None or not str(value).strip():
        raise ValueError(
            f"phoneme_language_missing: audio {audio.audio_file_id} ({audio.name!r}) "
            "has no catalog language"
        )
    return str(value).strip().lower().replace("_", "-")
