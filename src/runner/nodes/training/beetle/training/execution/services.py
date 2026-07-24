from dataclasses import dataclass

from ...data import ValidationLoader
from ..distributed import DistributedRuntime
from ..loop_events import TrainingLifecycle
from ..reporting import (
    MlflowSession,
    MAX_PENDING_ARTIFACT_JOBS,
    ReportingState,
    StepObservation,
    StepObservationTracker,
    SystemMetricsSampler,
    TrainingReporter,
    TrainingMetric,
    configured_mlflow_client,
)
from ..runtime import RunPreparation
from ..validation import (
    ArtifactQueue,
    ValidationRunner,
    ValidationArtifacts,
    ValidationCoordinator,
)


@dataclass(frozen=True)
class RuntimeServices:
    lifecycle: TrainingLifecycle
    reporting: StepObservationTracker
    validation: ValidationCoordinator


class _DistributedReporter:
    def __init__(
        self,
        runtime: DistributedRuntime,
        reporter: TrainingReporter | None,
    ) -> None:
        self.runtime = runtime
        self.reporter = reporter

    def publish(
        self,
        observation: StepObservation,
        state: ReportingState,
        validation_metrics: tuple[TrainingMetric, ...],
    ) -> ReportingState:
        updated = (
            self._main().publish(observation, state, validation_metrics)
            if self.runtime.is_main_process
            else None
        )
        shared = self.runtime.broadcast_object(updated)
        if not isinstance(shared, ReportingState):
            raise TypeError("main process did not broadcast reporting state")
        return shared

    def flush(self) -> None:
        if self.runtime.is_main_process:
            self._main().flush()
        self.runtime.wait_for_everyone()

    def finish(self, state: ReportingState) -> ReportingState:
        updated = self._main().finish(state) if self.runtime.is_main_process else None
        shared = self.runtime.broadcast_object(updated)
        if not isinstance(shared, ReportingState):
            raise TypeError("main process did not broadcast reporting state")
        return shared

    def fail(self) -> None:
        if self.runtime.is_main_process:
            self._main().fail()

    def _main(self) -> TrainingReporter:
        if self.reporter is None:
            raise RuntimeError("main process reporter is unavailable")
        return self.reporter


def build_runtime_services(
    preparation: RunPreparation,
    validator: ValidationRunner,
    phoneme_tokenizer: object,
    text_tokenizer: object,
    runtime: DistributedRuntime,
) -> RuntimeServices:
    training_config = preparation.config.training
    recordings = ValidationLoader(preparation.config).collate(
        preparation.validation,
        phoneme_tokenizer,
        text_tokenizer,
    )
    session, state = _reporting_session(preparation, runtime)
    try:
        reporter = None
        artifacts = None
        if runtime.is_main_process:
            if session is None:
                raise RuntimeError("main process MLflow session is unavailable")
            system = SystemMetricsSampler(runtime.device.index or 0)
            reporter = TrainingReporter(session, training_config.total_steps, system)
            artifacts = ValidationArtifacts(
                preparation.checkpoint_manager.root.parent,
                preparation.config.audio.sample_rate,
                session,
                ArtifactQueue(
                    workers=2,
                    capacity=MAX_PENDING_ARTIFACT_JOBS,
                ),
            )
        validation = ValidationCoordinator(
            validator,
            recordings,
            artifacts,
            runtime,
        )
        lifecycle = TrainingLifecycle(
            training_config.total_steps,
            training_config.validation_every_steps,
            _DistributedReporter(runtime, reporter),
            validation,
        )
    except Exception:
        if session is not None:
            session.fail()
        raise
    return RuntimeServices(lifecycle, StepObservationTracker(state), validation)


def _reporting_session(
    preparation: RunPreparation,
    runtime: DistributedRuntime,
) -> tuple[MlflowSession | None, ReportingState]:
    session = None
    state = preparation.resume.reporting if preparation.resume is not None else None
    if runtime.is_main_process:
        client = configured_mlflow_client()
        if state is None:
            session = MlflowSession.start(
                client,
                preparation.config.model_dump(mode="json"),
            )
            state = ReportingState.initial(session.run_id)
        else:
            session = MlflowSession.resume(client, state.require_started())
    shared = runtime.broadcast_object(state)
    if not isinstance(shared, ReportingState):
        raise TypeError("main process did not broadcast reporting state")
    return session, shared
