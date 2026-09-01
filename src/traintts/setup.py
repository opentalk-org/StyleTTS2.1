from dataclasses import dataclass
import logging
import warnings

import torch
from accelerate import Accelerator
from munch import Munch
from torch import nn

from .config import TrainingConfig
from .features import SpeakerFeatures, WavLMFeatures
from .loading import (
    build_model,
    load_ASR_models,
    load_checkpoint,
    load_F0_models,
)
from .losses import MultiResolutionSTFTLoss
from .modules.plbert import load_plbert
from .optimizers import MultiOptimizer, build_optimizer
from .utils import recursive_munch


logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    modules: Munch
    parameters: Munch
    n_down: int

    def set_training_mode(self, training_modules: set[str]) -> None:
        for module in self.modules.values():
            module.eval()
        eval_mode_optimizers = {"style_encoder", "predictor_encoder"}
        for name in training_modules - eval_mode_optimizers:
            self.modules[name].train()


@dataclass
class LossBundle:
    stft: nn.Module


@dataclass
class FeatureBundle:
    wavlm: WavLMFeatures
    speaker: SpeakerFeatures | None


@dataclass
class TrainingRuntime:
    accelerator: Accelerator
    models: ModelBundle
    losses: LossBundle
    features: FeatureBundle
    optimizer: MultiOptimizer
    initial_step: int


def build_training_runtime(
    config: TrainingConfig,
    accelerator: Accelerator,
) -> TrainingRuntime:

    device = accelerator.device
    parameters = recursive_munch(config.model_params)
    modules = _build_models(config, parameters, device)
    if device.type == "cuda":
        torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message="TensorFloat32 tensor cores for float32 matrix multiplication.*",
            category=UserWarning,
            module=r"torch\._inductor\.compile_fx",
        )
        generator = modules.decoder.generator
        for name in (
            "_pre_upsample_noise",
            "_forward_res_block",
            "_output_forward",
        ):
            setattr(
                generator,
                name,
                torch.compile(
                    getattr(generator, name),
                    dynamic=True,
                    mode="default",
                ),
            )
    optimizer = _build_optimizer(config, modules)
    modules, optimizer, initial_step = _load_base_checkpoint(
        config,
        modules,
        optimizer,
    )
    modules, optimizer = _prepare_training(
        accelerator,
        modules,
        optimizer,
    )
    model_bundle = ModelBundle(
        modules,
        parameters,
        accelerator.unwrap_model(modules.text_aligner).n_down,
    )
    losses = LossBundle(MultiResolutionSTFTLoss().to(device))
    wavlm = WavLMFeatures(
        parameters.slm.model,
        config.preprocess_params.sr,
        parameters.slm.sr,
    )
    speaker_required = any(
        stage.loss_weights.speaker_feature > 0
        or stage.loss_weights.speaker_similarity > 0
        for stage in config.training_stages
    )
    speaker = (
        SpeakerFeatures(config.preprocess_params.sr)
        if speaker_required
        else None
    )
    features = FeatureBundle(wavlm.to(device), speaker)
    if speaker is not None:
        speaker.to(device)
    torch.cuda.empty_cache()
    return TrainingRuntime(
        accelerator,
        model_bundle,
        losses,
        features,
        optimizer,
        initial_step,
    )

def build_accelerator(config: TrainingConfig) -> Accelerator:
    precision = {
        "fp32": "no",
        "fp16": "fp16",
        "bf16": "bf16",
    }[config.precision]
    accelerator = Accelerator(
        mixed_precision=precision,
        cpu=config.device == "cpu",
    )
    if accelerator.device.type != config.device:
        raise RuntimeError(
            f"requested {config.device}, Accelerate selected "
            f"{accelerator.device.type}"
        )
    if accelerator.num_processes != config.distributed_processes:
        raise RuntimeError(
            f"requested {config.distributed_processes} distributed processes, "
            f"Accelerate started {accelerator.num_processes}"
        )
    return accelerator


def _build_models(
    config: TrainingConfig,
    parameters: Munch,
    device: torch.device,
) -> Munch:
    aligner = load_ASR_models(config.ASR_path, config.ASR_config)
    pitch = load_F0_models(config.F0_path)
    plbert = load_plbert(
        config.PLBERT_path,
        config.PLBERT_config,
        config.pretrained_model,
    )
    modules = build_model(parameters, aligner, pitch, plbert)
    for module in modules.values():
        module.to(device)
    return modules


def _build_optimizer(
    config: TrainingConfig,
    modules: Munch,
) -> MultiOptimizer:
    settings = config.optimizer_params
    schedule = {
        "max_lr": settings.lr,
        "pct_start": 0.0,
        "total_steps": config.total_steps,
    }
    schedules = {name: schedule.copy() for name in modules}
    schedules["bert"]["max_lr"] = settings.bert_lr
    optimizer = build_optimizer(
        {name: module.parameters() for name, module in modules.items()},
        scheduler_params_dict=schedules,
        lr=settings.lr,
    )
    return optimizer


def _load_base_checkpoint(
    config: TrainingConfig,
    modules: Munch,
    optimizer: MultiOptimizer,
) -> tuple[Munch, MultiOptimizer, int]:
    if config.pretrained_model is None:
        return modules, optimizer, 0
    if config.load_only_params:
        logger.warning(
            "Checkpoint optimizer state will not be loaded; all optimizer "
            "state is being replaced with newly initialized state"
        )
    ignored: list[str] = []
    for path, name in (
        (config.ASR_path, "text_aligner"),
        (config.F0_path, "pitch_extractor"),
        (config.PLBERT_path, "bert"),
    ):
        if path is not None:
            ignored.append(name)
    modules, optimizer, initial_step = load_checkpoint(
        modules,
        optimizer,
        config.pretrained_model,
        load_only_params=config.load_only_params,
        ignore_modules=ignored,
    )
    if config.reset_training_step:
        initial_step = 0
    settings = config.optimizer_params
    for name, module_optimizer in optimizer.optimizers.items():
        learning_rate = settings.bert_lr if name == "bert" else settings.lr
        for group in module_optimizer.param_groups:
            for key in ("lr", "initial_lr", "max_lr", "min_lr"):
                if key in group:
                    group[key] = learning_rate
    return modules, optimizer, initial_step


def _prepare_training(
    accelerator: Accelerator,
    modules: Munch,
    optimizer: MultiOptimizer,
) -> tuple[Munch, MultiOptimizer]:
    optimizer_names = tuple(optimizer.optimizers.keys())
    scheduler_names = tuple(optimizer.schedulers.keys())
    arguments = [optimizer.optimizers[name] for name in optimizer_names]
    arguments.extend(optimizer.schedulers[name] for name in scheduler_names)
    prepared = accelerator.prepare(*arguments)
    optimizer_end = len(optimizer_names)
    for name, item in zip(
        optimizer_names,
        prepared[:optimizer_end],
        strict=True,
    ):
        optimizer.optimizers[name] = item
    for name, item in zip(
        scheduler_names,
        prepared[optimizer_end:],
        strict=True,
    ):
        optimizer.schedulers[name] = item
    return modules, optimizer
