from pathlib import Path

import torch

from ...models import (
    Stage2Dependencies,
    build_stage1_models,
    build_stage2_models,
    compile_stage1,
)
from ..callbacks import TrainingCallbacks
from ..distributed import DistributedCallbacks, DistributedRuntime
from ..runtime import (
    RunPreparation,
    load_aligner,
    load_f0_extractor,
    load_phoneme_resources,
    load_text_resources,
    prepare_run,
)
from ..stage1 import Stage1Trainer, build_stage1_optimizers
from ..stage2 import Stage2Trainer
from ..stage2_inputs import DefaultStage2InputBuilder
from ..stage2_setup import (
    build_latent_flow_ema,
    build_stage2_optimizer,
    build_stage3_optimizers,
)
from ..stage3 import Stage3Trainer
from ..state import LoopState, StageKind
from ..validation import (
    Stage1ValidationEvaluator,
    Stage2ValidationEvaluator,
    Stage3ValidationEvaluator,
    ValidationRuntime,
)
from .support import (
    DatabaseSpeakerIndex,
    IgnoredTokenizer,
    RuntimeCallbacks,
    dependency_payload,
    initial_loop,
    intervals,
    report_models,
    restore_stage1,
    restore_stage2,
    train,
)


def run_stage(
    stage: StageKind,
    config_path: Path,
    output_path: Path,
    resume_path: Path | None,
    callbacks: RuntimeCallbacks,
    stage1_checkpoint: Path | None = None,
    stage2_checkpoint: Path | None = None,
) -> LoopState:
    preparation = prepare_run(
        stage,
        config_path,
        output_path,
        resume_path,
        callbacks,
    )
    torch.set_num_threads(
        preparation.config.data.prefetch.preprocessing_threads
    )
    stage_config = {
        StageKind.STAGE1: preparation.config.stage1,
        StageKind.STAGE2: preparation.config.stage2,
        StageKind.STAGE3: preparation.config.stage3,
    }[stage]
    runtime = DistributedRuntime(
        stage_config.precision,
        torch.device(preparation.config.runtime.device),
    )
    distributed_callbacks = DistributedCallbacks(callbacks, runtime)
    distributed_callbacks.check_cancel()
    if stage is StageKind.STAGE1:
        return _run_stage1(preparation, distributed_callbacks, runtime)
    if stage is StageKind.STAGE2:
        if stage1_checkpoint is None:
            raise ValueError("Stage 2 requires a Stage 1 checkpoint")
        return _run_stage2(
            preparation, distributed_callbacks, runtime, stage1_checkpoint
        )
    if stage1_checkpoint is None or stage2_checkpoint is None:
        raise ValueError("Stage 3 requires Stage 1 and Stage 2 checkpoints")
    return _run_stage3(
        preparation,
        distributed_callbacks,
        runtime,
        stage1_checkpoint,
        stage2_checkpoint,
    )


def _run_stage1(
    preparation: RunPreparation,
    callbacks: TrainingCallbacks,
    runtime: DistributedRuntime,
) -> LoopState:
    config = preparation.config
    models = build_stage1_models(config, load_f0_extractor())
    report_models(
        models, None, config, runtime.device, retain_audio_path=True
    )
    if config.runtime.compile:
        compile_stage1(models)
    trainer = Stage1Trainer(
        models,
        config.stage1,
        config.adversarial,
        config.runtime.seed,
        runtime,
        build_stage1_optimizers(models, config.stage1, runtime),
        intervals(preparation),
        preparation.config_fingerprint,
        preparation.index.fingerprint,
        initial_loop(StageKind.STAGE1),
    )
    sampler = trainer.restore(preparation.resume) if preparation.resume else None
    return train(
        preparation,
        trainer,
        callbacks,
        IgnoredTokenizer(),
        IgnoredTokenizer(),
        sampler,
        ValidationRuntime(
            Stage1ValidationEvaluator(
                models,
                config.stage1,
                config.runtime.seed,
                runtime.device,
            )
        ),
        runtime,
    )


def _run_stage2(
    preparation: RunPreparation,
    callbacks: TrainingCallbacks,
    runtime: DistributedRuntime,
    stage1_checkpoint: Path,
) -> LoopState:
    config = preparation.config
    stage1 = build_stage1_models(config, load_f0_extractor())
    payload = dependency_payload(
        stage1_checkpoint,
        StageKind.STAGE1,
        preparation,
    )
    restore_stage1(payload, stage1)
    phoneme = load_phoneme_resources(Path(config.architecture.phoneme.model_path))
    text = load_text_resources(config.architecture.text_encoder.pretrained_model)
    models = build_stage2_models(
        config,
        stage1,
        Stage2Dependencies(phoneme.model, text.model, load_aligner(config)),
    )
    ema = build_latent_flow_ema(models)
    report_models(
        stage1, models, config, runtime.device, retain_audio_path=True
    )
    input_builder = _input_builder(preparation, runtime)
    trainer = Stage2Trainer(
        models,
        ema,
        config.stage2,
        runtime,
        build_stage2_optimizer(models, config.stage2, runtime),
        intervals(preparation),
        preparation.config_fingerprint,
        preparation.index.fingerprint,
        initial_loop(StageKind.STAGE2),
        input_builder,
    )
    sampler = trainer.restore(preparation.resume) if preparation.resume else None
    return train(
        preparation,
        trainer,
        callbacks,
        phoneme.tokenizer,
        text.tokenizer,
        sampler,
        ValidationRuntime(
            Stage2ValidationEvaluator(
                stage1,
                models,
                ema,
                input_builder,
                config.stage2,
                config.runtime.seed,
                runtime.device,
            )
        ),
        runtime,
    )


def _run_stage3(
    preparation: RunPreparation,
    callbacks: TrainingCallbacks,
    runtime: DistributedRuntime,
    stage1_checkpoint: Path,
    stage2_checkpoint: Path,
) -> LoopState:
    config = preparation.config
    stage1 = build_stage1_models(config, load_f0_extractor())
    first = dependency_payload(
        stage1_checkpoint,
        StageKind.STAGE1,
        preparation,
    )
    restore_stage1(first, stage1)
    phoneme = load_phoneme_resources(Path(config.architecture.phoneme.model_path))
    text = load_text_resources(config.architecture.text_encoder.pretrained_model)
    stage2 = build_stage2_models(
        config,
        stage1,
        Stage2Dependencies(phoneme.model, text.model, load_aligner(config)),
    )
    ema = build_latent_flow_ema(stage2)
    second = dependency_payload(
        stage2_checkpoint,
        StageKind.STAGE2,
        preparation,
    )
    restore_stage2(second, stage2, ema)
    report_models(
        stage1, stage2, config, runtime.device, retain_audio_path=True
    )
    input_builder = _input_builder(preparation, runtime)
    trainer = Stage3Trainer(
        stage1,
        stage2,
        ema,
        config.stage3,
        config.adversarial,
        config.runtime.seed,
        runtime,
        build_stage3_optimizers(stage1, stage2, config.stage3, runtime),
        intervals(preparation),
        preparation.config_fingerprint,
        preparation.index.fingerprint,
        initial_loop(StageKind.STAGE3),
        input_builder,
    )
    sampler = trainer.restore(preparation.resume) if preparation.resume else None
    return train(
        preparation,
        trainer,
        callbacks,
        phoneme.tokenizer,
        text.tokenizer,
        sampler,
        ValidationRuntime(
            Stage3ValidationEvaluator(
                stage1,
                stage2,
                ema,
                input_builder,
                config.stage3,
                config.runtime.seed,
                runtime.device,
            )
        ),
        runtime,
    )


def _input_builder(
    preparation: RunPreparation,
    runtime: DistributedRuntime,
) -> DefaultStage2InputBuilder:
    config = preparation.config
    speakers = DatabaseSpeakerIndex(
        preparation.index,
        config.architecture.embeddings.speaker_classes,
    )
    return DefaultStage2InputBuilder(config, speakers, runtime)
