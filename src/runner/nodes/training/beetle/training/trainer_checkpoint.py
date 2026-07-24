from torch import nn

from ..data.pipeline import DataPipelineState
from .checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointPayload,
    StateKind,
    StateTarget,
    capture_named_state,
    restore_named_states,
    validate_resume_fingerprints,
)
from .reporting import ReportingState
from .setup import named_trainable_conditional_modules
from .state import LoopState


def checkpoint_payload(
    trainer,
    loop: LoopState,
    sampler_state: DataPipelineState,
    reporting: ReportingState,
) -> CheckpointPayload:
    return CheckpointPayload(
        version=CHECKPOINT_VERSION,
        config_fingerprint=trainer.config_fingerprint,
        data_fingerprint=trainer.data_fingerprint,
        loop=loop,
        rank_states=(trainer.runtime.capture_rank_state(),),
        states=tuple(
            capture_named_state(name, kind, module)
            for name, kind, module in state_modules(trainer)
        )
        + trainer.optimizers.capture_states(),
        sampler_state=sampler_state,
        loss_schedule=trainer.schedules.state(loop.optimizer_step),
        reporting=reporting,
    )


def restore_trainer(
    trainer,
    payload: CheckpointPayload,
    reset_optimizers: bool,
) -> DataPipelineState:
    validate_resume_fingerprints(
        payload,
        trainer.config_fingerprint,
        trainer.data_fingerprint,
    )
    expected_schedule = trainer.schedules.state(payload.loop.optimizer_step)
    if payload.loss_schedule != expected_schedule:
        raise ValueError("loss schedule state does not match training configuration")
    targets = tuple(
        StateTarget(name, kind, module)
        for name, kind, module in state_modules(trainer)
    ) + trainer.optimizers.state_targets()
    states = payload.states
    if reset_optimizers:
        reset_kinds = (StateKind.OPTIMIZER, StateKind.SCHEDULER, StateKind.SCALER)
        states = tuple(state for state in states if state.kind not in reset_kinds)
        targets = tuple(target for target in targets if target.kind not in reset_kinds)
    restore_named_states(states, targets)
    if len(payload.rank_states) != trainer.runtime.world_size:
        raise ValueError("checkpoint world size does not match distributed runtime")
    trainer.runtime.restore_rank_state(payload.rank_states[trainer.runtime.rank])
    trainer.set_loop_state(payload.loop)
    return payload.sampler_state


def state_modules(trainer) -> tuple[tuple[str, StateKind, nn.Module], ...]:
    acoustic = trainer.acoustic
    conditional = trainer.conditional
    acoustic_modules = (
        ("audio_encoder", StateKind.MODEL, trainer.runtime.unwrap(acoustic.audio_encoder)),
        ("feature_linear", StateKind.MODEL, trainer.runtime.unwrap(acoustic.feature_linear)),
        ("decoder", StateKind.MODEL, trainer.runtime.unwrap(acoustic.decoder)),
        ("generator", StateKind.MODEL, trainer.runtime.unwrap(acoustic.generator)),
        ("f0_extractor", StateKind.FROZEN_MODEL, acoustic.f0_extractor),
        (
            "discriminators",
            StateKind.DISCRIMINATOR,
            trainer.runtime.unwrap(acoustic.discriminators),
        ),
    )
    conditional_modules = tuple(
        (name, StateKind.MODEL, trainer.runtime.unwrap(module))
        for name, module in named_trainable_conditional_modules(conditional)
    )
    helpers = (
        ("text_encoder", StateKind.FROZEN_MODEL, conditional.text_encoder),
        ("latent_flow", StateKind.EMA, trainer.ema_latent_flow),
    )
    return (*acoustic_modules, *conditional_modules, *helpers)
