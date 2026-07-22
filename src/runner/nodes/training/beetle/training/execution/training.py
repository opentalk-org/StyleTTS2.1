from pathlib import Path

import torch

from ...models import (
    ConditionalDependencies,
    build_acoustic_models,
    build_conditional_models,
    compile_acoustic,
)
from ..callbacks import TrainingCallbacks
from ..distributed import DistributedCallbacks, DistributedRuntime
from ..runtime import (
    load_aligner,
    load_f0_extractor,
    load_phoneme_resources,
    load_text_resources,
    prepare_run,
)
from ..setup import build_latent_flow_ema, build_optimizers
from ..conditional_inputs import DefaultConditionalInputBuilder
from ..state import LoopState
from ..trainer import BeetleTrainer
from ..validation import TrainingValidationEvaluator, ValidationRuntime
from .support import (
    DatabaseSpeakerIndex,
    RuntimeCallbacks,
    initial_loop,
    intervals,
    report_models,
    train,
)


def run_training(
    config_path: Path,
    output_path: Path,
    resume_path: Path | None,
    callbacks: RuntimeCallbacks,
) -> LoopState:
    preparation = prepare_run(config_path, output_path, resume_path, callbacks)
    config = preparation.config
    torch.set_num_threads(config.data.prefetch.preprocessing_threads)
    runtime = DistributedRuntime(
        config.training.precision,
        torch.device(config.runtime.device),
    )
    distributed_callbacks = DistributedCallbacks(callbacks, runtime)
    distributed_callbacks.check_cancel()
    return _run(preparation, distributed_callbacks, runtime)


def _run(preparation, callbacks: TrainingCallbacks, runtime: DistributedRuntime) -> LoopState:
    config = preparation.config
    acoustic = build_acoustic_models(config, load_f0_extractor())
    phoneme = load_phoneme_resources(Path(config.architecture.phoneme.model_path))
    text = load_text_resources(config.architecture.text_encoder.pretrained_model)
    conditional = build_conditional_models(
        config,
        acoustic,
        ConditionalDependencies(phoneme.model, text.model, load_aligner(config)),
    )
    if config.runtime.compile:
        compile_acoustic(acoustic)
    ema = build_latent_flow_ema(conditional)
    report_models(acoustic, conditional, config, runtime.device, retain_audio_path=True)
    input_builder = _input_builder(preparation, runtime)
    trainer = BeetleTrainer(
        acoustic,
        conditional,
        ema,
        config.training,
        config.adversarial,
        config.runtime.seed,
        runtime,
        build_optimizers(acoustic, conditional, config.training, runtime),
        intervals(preparation),
        preparation.config_fingerprint,
        preparation.index.fingerprint,
        initial_loop(),
        input_builder,
    )
    sampler = trainer.restore(preparation.resume) if preparation.resume else None
    evaluator = TrainingValidationEvaluator(
        acoustic,
        conditional,
        ema,
        input_builder,
        config.training,
        config.runtime.seed,
        runtime.device,
    )
    return train(
        preparation,
        trainer,
        callbacks,
        phoneme.tokenizer,
        text.tokenizer,
        sampler,
        ValidationRuntime(evaluator),
        runtime,
    )


def _input_builder(
    preparation,
    runtime: DistributedRuntime,
) -> DefaultConditionalInputBuilder:
    config = preparation.config
    speakers = DatabaseSpeakerIndex(
        preparation.index,
        config.architecture.embeddings.speaker_classes,
    )
    return DefaultConditionalInputBuilder(config, speakers, runtime)
