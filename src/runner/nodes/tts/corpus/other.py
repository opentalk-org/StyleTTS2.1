from __future__ import annotations

import asyncio
import base64
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.assets.checkpoints import resolve_checkpoint_ref
from runner.nodes.datatypes import AudioPort
from runner.nodes.tts.corpus.audio import other_corpus_audio
from runner.nodes.tts.corpus.models import OtherCorpusJob
from runner.nodes.tts.corpus.other_plan import (
    EXPECTED_JOBS,
    build_other_corpus_plan,
    registered_stream_languages,
)
from runner.nodes.tts.corpus.references import load_registered_references
from runner.nodes.tts.corpus.state import completed_source_keys
from runner.nodes.tts.engines import load_engine
from runner.nodes.tts.engines.base import EngineRuntime, EngineSynthesisRequest
from runner.nodes.tts.voices import CloneReference, TtsEngine, Voice


class OtherTtsEngine(str, Enum):
    CHATTERBOX = TtsEngine.CHATTERBOX.value
    F5_TTS = TtsEngine.F5_TTS.value
    ORPHEUS = TtsEngine.ORPHEUS.value
    DIA = TtsEngine.DIA.value
    FISH_SPEECH = TtsEngine.FISH_SPEECH.value
    RAON_OPENTTS = TtsEngine.RAON_OPENTTS.value


class OtherTtsCorpusSynthesisSettings(StrictSettings):
    engine: OtherTtsEngine
    corpus_dir: Path
    source_dataset_ids: tuple[UUID, UUID]
    dataset_id: UUID
    dataset_name: str
    checkpoint_id: UUID
    batch_size: int = Field(default=4, ge=1, le=16)
    max_jobs: int | None = Field(default=None, ge=1)


class OtherTtsCorpusSynthesisNode(Node):
    NODE_TYPE = "OtherTtsCorpusSynthesis"
    DESCRIPTION = "Synthesize registered FineWiki streams through one lifecycle-loaded cloning or preset TTS engine with stored-reference voices and durable resume keys."
    CATEGORY = "TTS"
    SETTINGS = OtherTtsCorpusSynthesisSettings
    IS_INPUT = True
    INPUTS: dict[str, Any] = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 30},
        keep_loaded=True,
        exclusive_group="accelerator",
    )
    QUEUE_MAX_SIZE = 64

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._jobs: tuple[OtherCorpusJob, ...] = ()
        self._voices: dict[str, Voice] = {}
        self._runtime: EngineRuntime | None = None
        self._cursor = 0
        self._initialized = False

    async def setup(self, context: Any) -> None:
        engine = TtsEngine(self.settings.engine.value)
        expected_dataset = f"tts_{engine.value}"
        if self.settings.dataset_name != expected_dataset:
            raise ValueError(
                f"{engine.value}: dataset name must be {expected_dataset}"
            )
        stream_languages = registered_stream_languages(
            self.settings.corpus_dir,
            engine,
        )
        references = await asyncio.to_thread(
            load_registered_references,
            self.settings.source_dataset_ids,
            stream_languages,
        )
        jobs = await asyncio.to_thread(
            build_other_corpus_plan,
            self.settings.corpus_dir,
            engine,
            references,
        )
        if self.settings.max_jobs is not None:
            jobs = jobs[:self.settings.max_jobs]
        completed = await asyncio.to_thread(
            completed_source_keys,
            self.settings.dataset_id,
            self.settings.dataset_name,
        )
        self._jobs = tuple(
            job for job in jobs if job.source_key not in completed
        )
        self._voices = {
            stream: Voice(
                engine=engine,
                preset=None,
                clone=CloneReference(
                    wav_base64=base64.b64encode(reference.wav_bytes).decode("ascii"),
                    sample_rate=reference.sample_rate,
                    transcript=reference.transcript,
                ),
            )
            for stream, reference in references.items()
        }
        checkpoint = await asyncio.to_thread(
            resolve_checkpoint_ref,
            str(self.settings.checkpoint_id),
            engine.value,
        )
        self._runtime = await asyncio.to_thread(
            load_engine,
            engine,
            checkpoint.path,
        )
        self._initialized = True

    def remaining_items(self, context: Any) -> int | None:
        if not self._initialized:
            if self.settings.max_jobs is not None:
                return self.settings.max_jobs
            return EXPECTED_JOBS[TtsEngine(self.settings.engine.value)]
        return len(self._jobs) - self._cursor

    async def execute(self, batch, context):
        if self._runtime is None:
            raise RuntimeError("other TTS corpus runtime is not loaded")
        end = min(
            len(self._jobs),
            self._cursor + self.settings.batch_size,
        )
        selected = self._jobs[self._cursor:end]
        requests = [
            EngineSynthesisRequest(
                job.text,
                self._voice(job),
                job.language,
            )
            for job in selected
        ]
        results = await asyncio.to_thread(
            self._runtime.synthesize_batch,
            requests,
            context.check_cancel,
        )
        if len(results) != len(selected):
            raise RuntimeError(
                f"{self.settings.engine.value}: synthesis returned "
                f"{len(results)} results for {len(selected)} jobs"
            )
        self._cursor = end
        await context.report_progress(
            self.id,
            self._cursor,
            len(self._jobs),
            f"{self.settings.engine.value} corpus "
            f"{self._cursor}/{len(self._jobs)}",
        )
        return [
            {
                "audio": other_corpus_audio(
                    job,
                    result.samples,
                    result.sample_rate,
                )
            }
            for job, result in zip(selected, results, strict=True)
        ]

    async def teardown(self, context: Any) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        self._voices = {}
        release_accelerator_memory()

    def _voice(self, job: OtherCorpusJob) -> Voice:
        if job.engine is TtsEngine.ORPHEUS:
            return Voice(job.engine, job.voice_id, None)
        return self._voices[job.stream_id]
