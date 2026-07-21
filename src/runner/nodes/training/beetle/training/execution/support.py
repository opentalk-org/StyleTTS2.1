import logging
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn

from ...data import (
    ContinuousBatchPlanner,
    DataPipelineState,
    DatabaseSegmentIndex,
    DistributedShard,
    Stage1WindowPlanner,
    build_data_pipeline,
    build_stage1_window_geometry,
)
from ...models import Stage1Models, Stage2Models
from ...models.complexity import profile_latent_audio, require_complexity_budget
from ..callbacks import TrainingCallbacks
from ..checkpoint import (
    CheckpointManager,
    CheckpointPayload,
    StateKind,
    StateTarget,
    restore_named_states,
    validate_resume_fingerprints,
)
from ..distributed.checkpoint import DistributedCheckpointManager
from ..distributed import DistributedRuntime
from ..loop import LoopIntervals, run_continuously
from ..reporting import ReportingCompletion
from ..runtime import RunPreparation
from ..stage2_inputs import SpeakerIndex
from ..stage2_setup import named_trainable_stage2_modules
from ..state import LoopState, StageKind, TrainingPhase
from ..validation import StageValidator
from .services import build_runtime_services

logger = logging.getLogger(__name__)


class RuntimeCallbacks(TrainingCallbacks, Protocol):
    def report_index_progress(self, scanned: int, total: int) -> None: ...


class IgnoredTokenizer:
    def encode(self, text: str) -> list[int]:
        del text
        return []


class DatabaseSpeakerIndex(SpeakerIndex):
    def __init__(self, index: DatabaseSegmentIndex, maximum_classes: int) -> None:
        voices = sorted(
            {item.voice_id for item in index.records.values() if item.voice_id is not None}
        )
        if len(voices) > maximum_classes:
            raise ValueError(
                f"database has {len(voices)} voices but model supports "
                f"{maximum_classes} speaker classes"
            )
        self.entries = tuple(voices)

    def resolve(
        self,
        voice_ids: tuple[str | None, ...],
        device: torch.device,
    ) -> Tensor:
        missing = tuple(voice for voice in voice_ids if voice not in self.entries)
        if missing:
            raise ValueError(f"batch contains unknown voice labels: {missing}")
        indices = tuple(self.entries.index(voice) for voice in voice_ids)
        return torch.tensor(indices, dtype=torch.long, device=device)


def train(
    preparation: RunPreparation,
    trainer,
    callbacks: RuntimeCallbacks,
    phoneme_tokenizer,
    text_tokenizer,
    state: DataPipelineState | None,
    validator: StageValidator,
    runtime: DistributedRuntime,
) -> LoopState:
    if (
        preparation.resume is not None
        and preparation.resume.reporting.completion is ReportingCompletion.FINISHED
    ):
        return trainer.loop_state()
    pipeline_state = state or initial_pipeline_state(
        preparation,
        trainer.stage,
        runtime.shard,
    )
    pipeline = build_data_pipeline(
        preparation.config,
        stage_number(trainer.stage),
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
            trainer.stage,
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
    stage: StageKind,
    shard: DistributedShard,
) -> DataPipelineState:
    config = preparation.config
    stage_config = {
        StageKind.STAGE1: config.stage1,
        StageKind.STAGE2: config.stage2,
        StageKind.STAGE3: config.stage3,
    }[stage]
    if stage is StageKind.STAGE1:
        planner = Stage1WindowPlanner(
            preparation.index,
            stage_config.batch_size,
            config.runtime.seed,
            shard,
            build_stage1_window_geometry(config),
        )
    else:
        planner = ContinuousBatchPlanner(
            preparation.index,
            stage_number(stage),
            stage_config.batch_size,
            config.data.sentence_probability,
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


def dependency_payload(
    path: Path,
    stage: StageKind,
    preparation: RunPreparation,
) -> CheckpointPayload:
    payload = CheckpointManager(path.parent).load(path)
    validate_resume_fingerprints(
        payload,
        stage,
        preparation.config_fingerprint,
        preparation.index.fingerprint,
    )
    return payload


def restore_stage1(payload: CheckpointPayload, models: Stage1Models) -> None:
    targets = (
        StateTarget("audio_encoder", StateKind.MODEL, models.audio_encoder),
        StateTarget("feature_linear", StateKind.MODEL, models.feature_linear),
        StateTarget("decoder", StateKind.MODEL, models.decoder),
        StateTarget("generator", StateKind.MODEL, models.generator),
        StateTarget("f0_extractor", StateKind.FROZEN_MODEL, models.f0_extractor),
        StateTarget(
            "discriminators",
            StateKind.DISCRIMINATOR,
            models.discriminators,
        ),
    )
    restore_targets(payload, targets)


def restore_stage2(
    payload: CheckpointPayload,
    models: Stage2Models,
    ema: nn.Module,
) -> None:
    trainable = tuple(
        StateTarget(name, StateKind.MODEL, module)
        for name, module in named_trainable_stage2_modules(models)
    )
    frozen = (
        StateTarget("audio_encoder", StateKind.FROZEN_MODEL, models.audio_encoder),
        StateTarget("f0_extractor", StateKind.FROZEN_MODEL, models.f0_extractor),
        StateTarget("text_encoder", StateKind.FROZEN_MODEL, models.text_encoder),
    )
    restore_targets(
        payload,
        (*trainable, *frozen, StateTarget("latent_flow", StateKind.EMA, ema)),
    )


def restore_targets(
    payload: CheckpointPayload,
    targets: tuple[StateTarget, ...],
) -> None:
    selected = tuple(
        state
        for target in targets
        for state in payload.states
        if state.name == target.name and state.kind is target.kind
    )
    restore_named_states(selected, targets)


def report_models(
    stage1: Stage1Models,
    stage2: Stage2Models | None,
    config,
    device: torch.device,
    retain_audio_path: bool,
) -> None:
    stage1_report = stage1.parameter_report()
    inference = (
        stage1_report.inference
        if stage2 is None
        else stage2.parameter_report(stage1_report.inference).inference
    )
    minimum = config.complexity.minimum_inference_parameters
    maximum = config.complexity.maximum_inference_parameters
    message = (
        f"Beetle inference parameter count: {inference:,} "
        f"(target {minimum:,}-{maximum:,})"
    )
    if stage2 is None:
        logger.info("partial Stage 1 %s", message)
    else:
        (logger.warning if inference < minimum or inference > maximum else logger.info)(
            message
        )
    stage1.to(device)
    complexity = profile_latent_audio(
        stage1.feature_linear,
        stage1.decoder,
        stage1.generator,
        config,
    )
    logger.info(
        "latent-to-audio complexity: %.6f GFLOPs/s",
        complexity.gflops_per_second,
    )
    require_complexity_budget(complexity, config.complexity)
    if not retain_audio_path:
        stage1.feature_linear.cpu()
        stage1.decoder.cpu()
        stage1.generator.cpu()
        stage1.discriminators.cpu()


def intervals(preparation: RunPreparation) -> LoopIntervals:
    return LoopIntervals(
        preparation.config.runtime.log_every_steps,
        preparation.config.checkpoint.every_steps,
    )


def initial_loop(stage: StageKind) -> LoopState:
    return LoopState(stage, 0, 0, TrainingPhase.READY, 0, 0, 0, ())


def stage_number(stage: StageKind) -> int:
    return {
        StageKind.STAGE1: 1,
        StageKind.STAGE2: 2,
        StageKind.STAGE3: 3,
    }[stage]
