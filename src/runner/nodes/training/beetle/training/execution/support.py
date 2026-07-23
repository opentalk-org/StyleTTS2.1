from typing import Protocol

import torch
from torch import Tensor

from ...data import (
    ContinuousBatchPlanner,
    DataPipelineState,
    DatabaseSegmentIndex,
    DistributedShard,
    build_data_pipeline,
)
from ..callbacks import TrainingCallbacks
from ..distributed.checkpoint import DistributedCheckpointManager
from ..distributed import DistributedRuntime
from ..loop import LoopIntervals, run_continuously
from ..reporting import ReportingCompletion
from ..runtime import RunPreparation
from ..conditional_inputs import SpeakerIndex
from ..state import LoopState, TrainingPhase
from ..validation import ValidationRunner
from .services import build_runtime_services

class RuntimeCallbacks(TrainingCallbacks, Protocol):
    def report_index_progress(self, scanned: int, total: int) -> None: ...


class IgnoredTokenizer:
    def encode(self, text: str) -> list[int]:
        del text
        return []


class DatabaseSpeakerIndex(SpeakerIndex):
    def __init__(self, index: DatabaseSegmentIndex, maximum_classes: int) -> None:
        voices = sorted(
            {item.speaker_id for item in index.records.values() if item.speaker_id is not None}
        )
        if len(voices) > maximum_classes:
            raise ValueError(
                f"database has {len(voices)} voices but model supports "
                f"{maximum_classes} speaker classes"
            )
        self.entries = tuple(voices)

    def resolve(
        self,
        speaker_ids: tuple[str | None, ...],
        device: torch.device,
    ) -> Tensor:
        missing = tuple(voice for voice in speaker_ids if voice not in self.entries)
        if missing:
            raise ValueError(f"batch contains unknown voice labels: {missing}")
        indices = tuple(self.entries.index(voice) for voice in speaker_ids)
        return torch.tensor(indices, dtype=torch.long, device=device)


def train(
    preparation: RunPreparation,
    trainer,
    callbacks: RuntimeCallbacks,
    phoneme_tokenizer,
    text_tokenizer,
    state: DataPipelineState | None,
    validator: ValidationRunner,
    runtime: DistributedRuntime,
) -> LoopState:
    if (
        preparation.resume is not None
        and preparation.resume.reporting.completion is ReportingCompletion.FINISHED
    ):
        return trainer.loop_state()
    pipeline_state = state or initial_pipeline_state(
        preparation,
        runtime.shard,
    )
    pipeline = build_data_pipeline(
        preparation.config,
        callbacks,
        preparation.index,
        phoneme_tokenizer,
        text_tokenizer,
        pipeline_state,
        runtime.shard,
    )
    try:
        services = build_runtime_services(
            preparation,
            validator,
            phoneme_tokenizer,
            text_tokenizer,
            runtime,
        )
        try:
            return run_continuously(
                pipeline,
                trainer,
                callbacks,
                DistributedCheckpointManager(
                    preparation.checkpoint_manager,
                    runtime,
                ),
                services.reporting,
                services.lifecycle,
            )
        finally:
            services.validation.close()
    finally:
        pipeline.close()


def initial_pipeline_state(
    preparation: RunPreparation,
    shard: DistributedShard,
) -> DataPipelineState:
    config = preparation.config
    planner = ContinuousBatchPlanner(
        preparation.index,
        config.training.batch_size,
        config.runtime.seed,
        config.data.maximum_seconds,
        config.data.grouping,
        shard,
    )
    return DataPipelineState(
        preparation.index.fingerprint,
        planner.state_dict(),
        shard.world_size,
    )


def intervals(preparation: RunPreparation) -> LoopIntervals:
    return LoopIntervals(
        preparation.config.runtime.log_every_steps,
        preparation.config.checkpoint.every_steps,
    )


def initial_loop() -> LoopState:
    return LoopState(0, 0, TrainingPhase.READY, 0, 0, 0, ())
