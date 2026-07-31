from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from munch import Munch
from torch import nn

from .config import TrainingConfig
from .loading import (
    build_model,
    load_ASR_models,
    load_checkpoint,
    load_F0_models,
)
from .losses import (
    DiscriminatorLoss,
    GeneratorLoss,
    MultiResolutionSTFTLoss,
    WavLMLoss,
)
from .modules.diffusion.sampler import (
    ADPM2Sampler,
    DiffusionSampler,
    KarrasSchedule,
)
from .modules.plbert import load_plbert
from .modules.slmadv import SLMAdversarialLoss
from .optimizers import MultiOptimizer, build_optimizer
from .utils import recursive_munch


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
    generator: nn.Module
    discriminator: nn.Module
    wavlm: nn.Module
    stft: nn.Module
    slm_adversarial: SLMAdversarialLoss


@dataclass
class TrainingRuntime:
    accelerator: Accelerator
    models: ModelBundle
    losses: LossBundle
    optimizer: MultiOptimizer
    diffusion_sampler: DiffusionSampler


def build_training_runtime(
    config: TrainingConfig,
    accelerator: Accelerator,
) -> TrainingRuntime:

    device = accelerator.device
    parameters = recursive_munch(config.model_params)
    modules = _build_models(config, parameters, device)
    optimizer = _build_optimizer(config, modules)
    modules, optimizer = _load_base_checkpoint(config, modules, optimizer)
    if not config.profiling_enabled:
        modules.text_aligner.asr_s2s = torch.jit.script(
            modules.text_aligner.asr_s2s
        )
    modules, optimizer = _prepare_training(
        accelerator,
        modules,
        optimizer,
    )
    diffusion = accelerator.unwrap_model(modules.diffusion).diffusion
    sampler = DiffusionSampler(
        diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001,
            sigma_max=3.0,
            rho=9.0,
        ),
        clamp=False,
    )
    generator = GeneratorLoss(
        accelerator.unwrap_model(modules.mpd),
        accelerator.unwrap_model(modules.msd),
    ).to(device)
    discriminator = DiscriminatorLoss(modules.mpd, modules.msd).to(device)
    wavlm = WavLMLoss(
        parameters.slm.model,
        modules.wd,
        config.preprocess_params.sr,
        parameters.slm.sr,
    ).to(device)
    model_bundle = ModelBundle(
        modules,
        parameters,
        accelerator.unwrap_model(modules.text_aligner).n_down,
    )
    losses = LossBundle(
        generator,
        discriminator,
        wavlm,
        MultiResolutionSTFTLoss().to(device),
        _build_slm_loss(
            config,
            model_bundle,
            wavlm,
            sampler,
        ),
    )
    torch.cuda.empty_cache()
    return TrainingRuntime(
        accelerator,
        model_bundle,
        losses,
        optimizer,
        sampler,
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
    plbert = load_plbert(config.PLBERT_path, config.PLBERT_config)
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
    schedules["bert"]["max_lr"] = settings.bert_lr * 2
    schedules["decoder"]["max_lr"] = settings.ft_lr * 2
    schedules["style_encoder"]["max_lr"] = settings.ft_lr * 2
    optimizer = build_optimizer(
        {name: module.parameters() for name, module in modules.items()},
        scheduler_params_dict=schedules,
        lr=settings.lr,
    )
    _configure_optimizer_groups(optimizer, settings)
    return optimizer


def _configure_optimizer_groups(
    optimizer: MultiOptimizer,
    settings: Any,
) -> None:
    for group in optimizer.optimizers["bert"].param_groups:
        group.update(
            betas=(0.9, 0.99),
            lr=settings.bert_lr,
            initial_lr=settings.bert_lr,
            min_lr=0,
            weight_decay=0.01,
        )
    for name in ("decoder", "style_encoder"):
        for group in optimizer.optimizers[name].param_groups:
            group.update(
                betas=(0.0, 0.99),
                lr=settings.ft_lr,
                initial_lr=settings.ft_lr,
                min_lr=0,
                weight_decay=1e-4,
            )


def _load_base_checkpoint(
    config: TrainingConfig,
    modules: Munch,
    optimizer: MultiOptimizer,
) -> tuple[Munch, MultiOptimizer]:
    if config.pretrained_model is None:
        return modules, optimizer
    ignored = []
    for path, name in (
        (config.ASR_path, "text_aligner"),
        (config.F0_path, "pitch_extractor"),
        (config.PLBERT_path, "bert"),
    ):
        if path is not None:
            ignored.append(name)
    return load_checkpoint(
        modules,
        optimizer,
        config.pretrained_model,
        load_only_params=config.load_only_params,
        ignore_modules=ignored,
    )


def _build_slm_loss(
    config: TrainingConfig,
    models: ModelBundle,
    wavlm: nn.Module,
    sampler: DiffusionSampler,
) -> SLMAdversarialLoss:
    settings = config.slmadv_params
    return SLMAdversarialLoss(
        models.modules,
        wavlm,
        sampler,
        settings.min_len,
        settings.max_len,
        settings.batch_max_samples,
        skip_update=settings.iter,
        sig=settings.sig,
    )


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
