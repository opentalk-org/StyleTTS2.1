from dataclasses import dataclass

import torch

from ...data import ValidationLoader
from ..loop_events import TrainingLifecycle
from ..reporting import (
    MlflowSession,
    MAX_PENDING_ARTIFACT_JOBS,
    ReportingState,
    StepObservationTracker,
    SystemMetricsSampler,
    TrainingReporter,
    configured_mlflow_client,
)
from ..runtime import RunPreparation
from ..state import StageKind
from ..validation import (
    ArtifactQueue,
    StageValidator,
    ValidationArtifacts,
    ValidationCoordinator,
)


@dataclass(frozen=True)
class RuntimeServices:
    lifecycle: TrainingLifecycle
    reporting: StepObservationTracker
    validation: ValidationCoordinator


def build_runtime_services(
    preparation: RunPreparation,
    stage: StageKind,
    validator: StageValidator,
    phoneme_tokenizer: object,
    text_tokenizer: object,
) -> RuntimeServices:
    stage_config = {
        StageKind.STAGE1: preparation.config.stage1,
        StageKind.STAGE2: preparation.config.stage2,
        StageKind.STAGE3: preparation.config.stage3,
    }[stage]
    recordings = ValidationLoader.from_database(preparation.config).collate(
        preparation.validation,
        phoneme_tokenizer,
        text_tokenizer,
    )
    system = SystemMetricsSampler(
        torch.device(preparation.config.runtime.device).index or 0
    )
    session, state = _reporting_session(preparation, stage)
    try:
        reporter = TrainingReporter(session, stage_config.total_steps, system)
        validation = ValidationCoordinator(
            validator,
            recordings,
            ValidationArtifacts(
                preparation.checkpoint_manager.root.parent,
                preparation.config.audio.sample_rate,
                session,
                ArtifactQueue(
                    workers=2,
                    capacity=MAX_PENDING_ARTIFACT_JOBS,
                ),
            ),
        )
        lifecycle = TrainingLifecycle(
            stage_config.total_steps,
            stage_config.validation_every_steps,
            reporter,
            validation,
        )
    except Exception:
        session.fail()
        raise
    return RuntimeServices(lifecycle, StepObservationTracker(state), validation)


def _reporting_session(
    preparation: RunPreparation,
    stage: StageKind,
) -> tuple[MlflowSession, ReportingState]:
    client = configured_mlflow_client()
    if preparation.resume is None:
        session = MlflowSession.start(
            client,
            stage,
            preparation.config.model_dump(mode="json"),
        )
        return session, ReportingState.initial(session.run_id)
    state = preparation.resume.reporting
    session = MlflowSession.resume(client, state.require_started(), stage)
    return session, state
