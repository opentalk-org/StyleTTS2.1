from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.assets.checkpoints import resolve_checkpoint_ref
from runner.nodes.datatypes import AudioPort
from runner.nodes.tts.corpus.audio import corpus_audio
from runner.nodes.tts.corpus.models import CorpusJob
from runner.nodes.tts.corpus.plan import (
    EXPECTED_LINES,
    EXPECTED_PIPER_JOBS,
    build_corpus_plan,
    without_completed,
)
from runner.nodes.tts.corpus.state import completed_source_keys
from runner.nodes.tts.engines import load_engine
from runner.nodes.tts.engines.base import (
    EngineRuntime,
    EngineSynthesisRequest,
)
from runner.nodes.tts.piper_catalog import fetch_piper_catalog
from runner.nodes.tts.voices import TtsEngine, Voice


class KokoroCorpusSynthesisSettings(StrictSettings):
    corpus_dir: Path
    dataset_id: UUID
    dataset_name: Literal["tts_kokoro"] = "tts_kokoro"
    checkpoint_id: UUID
    batch_size: int = Field(default=16, ge=1, le=64)
    max_jobs: int | None = Field(default=None, ge=1)


class KokoroCorpusSynthesisNode(Node):
    NODE_TYPE = "KokoroCorpusSynthesis"
    DESCRIPTION = "Synthesize the Kokoro-assigned FineWiki corpus with one lifecycle model and resumable dataset keys."
    CATEGORY = "TTS"
    SETTINGS = KokoroCorpusSynthesisSettings
    IS_INPUT = True
    INPUTS: dict[str, Any] = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 2},
        keep_loaded=True,
        exclusive_group="accelerator",
    )
    QUEUE_MAX_SIZE = 128

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._jobs: tuple[CorpusJob, ...] = ()
        self._cursor = 0
        self._initialized = False
        self._runtime: EngineRuntime | None = None

    async def setup(self, context: Any) -> None:
        catalog = await asyncio.to_thread(fetch_piper_catalog)
        plan = await asyncio.to_thread(
            build_corpus_plan,
            self.settings.corpus_dir,
            catalog,
        )
        jobs = plan.kokoro_jobs
        if self.settings.max_jobs is not None:
            jobs = jobs[:self.settings.max_jobs]
        completed = await asyncio.to_thread(
            completed_source_keys,
            self.settings.dataset_id,
            self.settings.dataset_name,
        )
        self._jobs = without_completed(jobs, completed)
        checkpoint = await asyncio.to_thread(
            resolve_checkpoint_ref,
            str(self.settings.checkpoint_id),
            TtsEngine.KOKORO.value,
        )
        self._runtime = await asyncio.to_thread(
            load_engine,
            TtsEngine.KOKORO,
            checkpoint.path,
        )
        self._initialized = True

    def remaining_items(self, context: Any) -> int | None:
        if not self._initialized:
            return self.settings.max_jobs or (
                EXPECTED_LINES - EXPECTED_PIPER_JOBS
            )
        return len(self._jobs) - self._cursor

    async def execute(self, batch, context):
        if self._runtime is None:
            raise RuntimeError("Kokoro corpus runtime is not loaded")
        end = min(
            len(self._jobs),
            self._cursor + self.settings.batch_size,
        )
        selected = self._jobs[self._cursor:end]
        requests = [
            EngineSynthesisRequest(
                job.text,
                Voice(TtsEngine.KOKORO, job.voice_id, None),
                job.language,
            )
            for job in selected
        ]
        results = await asyncio.to_thread(
            self._runtime.synthesize_batch,
            requests,
            context.check_cancel,
        )
        self._cursor = end
        await context.report_progress(
            self.id,
            self._cursor,
            len(self._jobs),
            f"kokoro corpus {self._cursor}/{len(self._jobs)}",
        )
        return [
            {"audio": corpus_audio(job, result.samples, result.sample_rate)}
            for job, result in zip(selected, results, strict=True)
        ]

    async def teardown(self, context: Any) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        release_accelerator_memory()
