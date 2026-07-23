from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio
from runner.nodes.tts.corpus.audio import corpus_audio
from runner.nodes.tts.corpus.models import CorpusJob
from runner.nodes.tts.corpus.plan import (
    EXPECTED_PIPER_JOBS,
    build_corpus_plan,
    without_completed,
)
from runner.nodes.tts.corpus.state import completed_source_keys
from runner.nodes.tts.engines.piper import (
    PiperRuntime,
    PiperSynthesisOptions,
)
from runner.nodes.tts.piper_catalog import fetch_piper_catalog
from runner.nodes.tts.piper_download import download_piper_voice


class PiperCorpusSynthesisSettings(StrictSettings):
    corpus_dir: Path
    dataset_id: UUID
    dataset_name: Literal["tts_piper"] = "tts_piper"
    workers: int = Field(default=15, ge=1, le=32)
    jobs_per_worker: int = Field(default=8, ge=1, le=64)
    max_jobs: int | None = Field(default=None, ge=1)


@dataclass(slots=True)
class PiperShard:
    jobs: tuple[CorpusJob, ...]
    model_paths: dict[str, Path]
    jobs_per_worker: int
    cursor: int = 0
    loaded_voice_id: str | None = None
    runtime: PiperRuntime | None = None

    @property
    def remaining(self) -> int:
        return len(self.jobs) - self.cursor

    def synthesize_next(self, check_cancel) -> list[Audio]:
        if self.cursor == len(self.jobs):
            return []
        first = self.jobs[self.cursor]
        end = self.cursor + 1
        limit = min(len(self.jobs), self.cursor + self.jobs_per_worker)
        while end < limit and _same_runtime_group(first, self.jobs[end]):
            end += 1
        selected = self.jobs[self.cursor:end]
        self.cursor = end
        if self.loaded_voice_id != first.voice_id:
            self.runtime = PiperRuntime(
                self.model_paths[first.voice_id],
                threads=1,
            )
            self.loaded_voice_id = first.voice_id
        if self.runtime is None:
            raise RuntimeError("Piper shard runtime was not loaded")
        options = PiperSynthesisOptions(
            first.speaker_id,
            1.0,
            0.667,
            0.8,
            1.0,
        )
        results = self.runtime.synthesize_many(
            [job.text for job in selected],
            options,
            check_cancel,
        )
        return [
            corpus_audio(job, samples, sample_rate)
            for job, (samples, sample_rate) in zip(
                selected,
                results,
                strict=True,
            )
        ]


class PiperCorpusSynthesisNode(Node):
    NODE_TYPE = "PiperCorpusSynthesis"
    DESCRIPTION = "Synthesize the Piper-assigned FineWiki corpus with balanced one-thread ONNX workers and resumable dataset keys."
    CATEGORY = "TTS"
    SETTINGS = PiperCorpusSynthesisSettings
    IS_INPUT = True
    INPUTS: dict[str, Any] = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(
        resources={"cpu_workers": 15},
        keep_loaded=True,
    )
    QUEUE_MAX_SIZE = 256

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._shards: tuple[PiperShard, ...] = ()
        self._executor: ThreadPoolExecutor | None = None
        self._initialized = False
        self._total = 0
        self._completed = 0

    async def setup(self, context: Any) -> None:
        catalog = await asyncio.to_thread(fetch_piper_catalog)
        plan = await asyncio.to_thread(
            build_corpus_plan,
            self.settings.corpus_dir,
            catalog,
        )
        jobs = plan.piper_jobs
        if self.settings.max_jobs is not None:
            jobs = jobs[:self.settings.max_jobs]
        completed = await asyncio.to_thread(
            completed_source_keys,
            self.settings.dataset_id,
            self.settings.dataset_name,
        )
        pending = without_completed(jobs, completed)
        voice_ids = {job.voice_id for job in pending}
        selected = {
            voice.voice_id: voice
            for voice in catalog
            if voice.voice_id in voice_ids
        }
        checkpoints = await asyncio.gather(
            *(
                asyncio.to_thread(download_piper_voice, selected[voice_id])
                for voice_id in sorted(voice_ids)
            )
        )
        model_paths = {
            voice_id: checkpoint.path
            for voice_id, checkpoint in zip(
                sorted(voice_ids),
                checkpoints,
                strict=True,
            )
        }
        shards = shard_piper_jobs(pending, self.settings.workers)
        self._shards = tuple(
            PiperShard(shard, model_paths, self.settings.jobs_per_worker)
            for shard in shards
            if shard
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(self._shards)),
            thread_name_prefix="piper-corpus",
        )
        self._total = len(pending)
        self._initialized = True

    def remaining_items(self, context: Any) -> int | None:
        if not self._initialized:
            return self.settings.max_jobs or EXPECTED_PIPER_JOBS
        return sum(shard.remaining for shard in self._shards)

    async def execute(self, batch, context):
        if self._executor is None:
            raise RuntimeError("Piper corpus executor is not loaded")
        loop = asyncio.get_running_loop()
        active = [shard for shard in self._shards if shard.remaining]
        groups = await asyncio.gather(
            *(
                loop.run_in_executor(
                    self._executor,
                    shard.synthesize_next,
                    context.check_cancel,
                )
                for shard in active
            )
        )
        audios = [audio for group in groups for audio in group]
        self._completed += len(audios)
        await context.report_progress(
            self.id,
            self._completed,
            self._total,
            f"piper corpus {self._completed}/{self._total}",
        )
        return [{"audio": audio} for audio in audios]

    async def teardown(self, context: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None
        self._shards = ()


def shard_piper_jobs(
    jobs: tuple[CorpusJob, ...],
    worker_count: int,
) -> tuple[tuple[CorpusJob, ...], ...]:
    if worker_count < 1:
        raise ValueError("Piper corpus worker count must be positive")
    streams: defaultdict[str, list[CorpusJob]] = defaultdict(list)
    for job in jobs:
        streams[job.stream_id].append(job)
    shards: list[list[CorpusJob]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    groups = sorted(
        streams.values(),
        key=lambda group: (-len(group), group[0].stream_id),
    )
    for group in groups:
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        shards[worker].extend(group)
        loads[worker] += len(group)
    return tuple(
        tuple(sorted(shard, key=_job_group_key))
        for shard in shards
    )


def _job_group_key(job: CorpusJob) -> tuple[str, int, str, int]:
    return (
        job.voice_id,
        -1 if job.speaker_id is None else job.speaker_id,
        job.stream_id,
        job.sentence_index,
    )


def _same_runtime_group(first: CorpusJob, second: CorpusJob) -> bool:
    return (
        first.voice_id == second.voice_id
        and first.speaker_id == second.speaker_id
    )
