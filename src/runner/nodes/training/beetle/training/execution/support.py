from typing import Protocol

import torch
from torch import Tensor

from ...data import (
    ContinuousBatchPlanner,
    DataPipelineState,
    DatabaseSegmentIndex,
    DistributedShard,
    RepeatedBatchPipeline,
    ValidationLoader,
    build_data_pipeline,
    repeat_validation_embedding_groups,
)
from ..callbacks import TrainingCallbacks
from ..conditional.input_types import SpeakerIndex
from ..distributed import DistributedRuntime
from ..distributed.checkpoint import DistributedCheckpointManager
from ..loop import LoopIntervals, run_continuously
from ..reporting import ReportingCompletion
from ..runtime import RunPreparation
from ..state import LoopState, TrainingPhase
from ..validation import ValidationRunner
from .services import build_runtime_services


class RuntimeCallbacks(TrainingCallbacks, Protocol):
    def report_index_progress(self, scanned: int, total: int) -> None: ...


class DatabaseSpeakerIndex(SpeakerIndex):
    def __init__(self, index: DatabaseSegmentIndex, maximum_classes: int) -> None:
        voices = sorted(
            {
                item.speaker_id
                for item in index.records.values()
                if item.speaker_id is not None
            }
            | set(index.validation.conditional_by_voice)
        )
        self.entries = tuple(voices)

    def resolve(
        self,
        speaker_ids: tuple[str | None, ...],
        device: torch.device,
    ) -> Tensor:
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
    recordings = ValidationLoader(preparation.config).collate(
        preparation.validation,
        phoneme_tokenizer,
        text_tokenizer,
    )
    if preparation.config.training.overfit_validation_recording:
        recordings = (repeat_validation_embedding_groups(recordings[0]),)
        pipeline = RepeatedBatchPipeline(
            recordings[0].batch,
            preparation.index.fingerprint,
            runtime.world_size,
            state,
        )
    else:
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
            recordings,
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
