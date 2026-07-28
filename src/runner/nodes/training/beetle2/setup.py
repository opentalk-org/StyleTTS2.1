import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AlbertConfig, AlbertModel, BertModel, BertTokenizerFast

from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session

from .config import BeetleConfig, OptimizerConfig, TrainingStage
from .models import (
    AcousticModels,
    ConditionalDependencies,
    ConditionalModels,
    F0Extractor,
    build_acoustic_models,
    build_conditional_models,
    compile_acoustic,
)
from .models.modules.aligner import PhonemeAligner
from .models.modules.alignment_backbone import StyleTTSAlignerBackbone
from .models.modules.audio import AudioEncoder
from .models.modules.decoder import Decoder
from .models.modules.generator import Generator
from .phonemes import PhonemeTokenizer


@dataclass(frozen=True)
class TextResources:
    phoneme_tokenizer: PhonemeTokenizer
    text_tokenizer: BertTokenizerFast
    phoneme_model: AlbertModel | None
    text_model: BertModel | None


@dataclass
class TrainingOptimizers:
    generator: torch.optim.AdamW
    discriminator: torch.optim.AdamW | None


class FrozenAcoustic(nn.Module):
    def __init__(
        self,
        audio_encoder: AudioEncoder,
        f0_extractor: F0Extractor,
        decoder: Decoder,
        generator: Generator,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.f0_extractor = f0_extractor
        self.decoder = decoder
        self.generator = generator


@dataclass
class TrainingModels:
    acoustic: AcousticModels | None
    conditional: ConditionalModels | None
    latent_flow_ema: nn.Module | None


def load_text_resources(config: BeetleConfig) -> TextResources:
    stage = config.training.stage
    text_snapshot = Path(
        snapshot_download(
            config.architecture.text_encoder.pretrained_model,
            local_files_only=True,
        )
    )
    text_tokenizer = BertTokenizerFast.from_pretrained(text_snapshot)
    phoneme_model = None
    text_model = None
    if stage is not TrainingStage.POSTERIOR:
        model_path = Path(config.architecture.phoneme.model_path)
        settings = yaml.safe_load((model_path / "config.yml").read_text())
        phoneme_model = AlbertModel(AlbertConfig(**settings["model_params"]))
        checkpoints = sorted(
            model_path.glob("step_*.t7"),
            key=lambda path: int(path.stem.removeprefix("step_")),
        )
        if not checkpoints:
            raise ValueError(f"PL-BERT checkpoint is missing from {model_path}")
        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        prefix = "module.encoder."
        state = {
            key.removeprefix(prefix): value
            for key, value in payload["net"].items()
            if key.startswith(prefix) and not key.endswith("position_ids")
        }
        incompatible = phoneme_model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"PL-BERT checkpoint keys do not match: {incompatible}")
        text_model = BertModel.from_pretrained(text_snapshot)
    return TextResources(
        PhonemeTokenizer(),
        text_tokenizer,
        phoneme_model,
        text_model,
    )


def load_f0_extractor() -> F0Extractor:
    checkpoint = (
        Path(__file__).parent.parent
        / "hiftnet"
        / "Utils"
        / "JDC"
        / "bst.t7"
    )
    return F0Extractor.from_checkpoint(checkpoint)


def load_aligner(config: BeetleConfig) -> PhonemeAligner:
    settings = config.architecture.aligner
    with database_session() as session:
        folder = asset_crud.get_checkpoint_path(session, settings.checkpoint_asset_id)
    checkpoint = folder / settings.checkpoint_filename
    backbone = StyleTTSAlignerBackbone(
        input_channels=config.audio.mel_channels,
        hidden_channels=settings.hidden_channels,
        token_count=config.architecture.phoneme_token_count,
        layer_count=settings.layer_count,
        token_embedding_channels=settings.token_embedding_channels,
    )
    aligner = PhonemeAligner(
        backbone,
        settings,
        config.architecture.phoneme_token_count,
        settings.frame_reduction,
    )
    aligner.load_checkpoint(checkpoint)
    return aligner


def build_models(
    config: BeetleConfig,
    resources: TextResources,
) -> TrainingModels:
    stage = config.training.stage
    f0_extractor = load_f0_extractor()
    acoustic = None
    conditional = None
    if stage in (TrainingStage.POSTERIOR, TrainingStage.END_TO_END):
        acoustic = build_acoustic_models(config, f0_extractor)
    if stage is not TrainingStage.POSTERIOR:
        if resources.phoneme_model is None or resources.text_model is None:
            raise RuntimeError("conditional text models were not loaded")
        acoustic_dependency: nn.Module
        if acoustic is None:
            acoustic_dependency = FrozenAcoustic(
                AudioEncoder(config.architecture.posterior),
                f0_extractor,
                Decoder(config.architecture.decoder),
                Generator(
                    config.architecture.generator,
                    config.audio.sample_rate,
                ),
            ).requires_grad_(False)
        else:
            acoustic_dependency = acoustic
        conditional = build_conditional_models(
            config,
            acoustic_dependency,
            ConditionalDependencies(
                resources.phoneme_model,
                resources.text_model,
                load_aligner(config),
            ),
        )
    ema = (
        copy.deepcopy(conditional.latent_flow).requires_grad_(False).eval()
        if conditional is not None
        else None
    )
    if config.runtime.compile and acoustic is not None:
        compile_acoustic(acoustic)
    return TrainingModels(acoustic, conditional, ema)


def trainable_conditional_modules(
    models: ConditionalModels,
) -> tuple[nn.Module, ...]:
    return (
        models.phoneme_encoder,
        models.latent_phoneme_encoder,
        models.duration_phoneme_encoder,
        models.context_phoneme_encoder,
        models.context_audio_encoder,
        models.style_encoder,
        models.voice_encoder,
        models.language_embedding,
        models.condition_bank,
        models.duration_predictor,
        models.latent_flow,
        models.aligner,
        models.style_speaker_classifier,
        models.style_statistics_head,
        models.voice_ge2e,
        models.style_ge2e,
    )


def build_optimizers(
    config: BeetleConfig,
    models: TrainingModels,
) -> TrainingOptimizers:
    generator_modules: list[nn.Module] = []
    if models.acoustic is not None:
        generator_modules.extend(
            (
                models.acoustic.audio_encoder,
                models.acoustic.feature_linear,
                models.acoustic.decoder,
                models.acoustic.generator,
            )
        )
    flow_parameters: tuple[nn.Parameter, ...] = ()
    if models.conditional is not None:
        generator_modules.extend(trainable_conditional_modules(models.conditional))
        flow_parameters = tuple(models.conditional.latent_flow.parameters())
    parameters = tuple(
        parameter
        for module in generator_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    flow_ids = {id(parameter) for parameter in flow_parameters}
    regular = tuple(parameter for parameter in parameters if id(parameter) not in flow_ids)
    settings = config.training.generator_optimizer
    groups: list[dict[str, object]] = [
        {"params": regular, "weight_decay": settings.weight_decay}
    ]
    if flow_parameters:
        groups.append(
            {
                "params": flow_parameters,
                "weight_decay": config.training.latent_flow_weight_decay,
            }
        )
    generator = torch.optim.AdamW(
        groups,
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
    )
    discriminator = None
    if models.acoustic is not None:
        discriminator = adamw(
            tuple(models.acoustic.discriminators.parameters()),
            config.training.discriminator_optimizer,
        )
    return TrainingOptimizers(generator, discriminator)


def adamw(
    parameters: tuple[nn.Parameter, ...],
    settings: OptimizerConfig,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
    )


def prepare_training(
    accelerator: Accelerator,
    models: TrainingModels,
    optimizers: TrainingOptimizers,
) -> tuple[TrainingModels, TrainingOptimizers]:
    acoustic = models.acoustic
    conditional = models.conditional
    entries: list[tuple[nn.Module, str, nn.Module]] = []
    if acoustic is not None:
        acoustic.to(accelerator.device)
        acoustic.f0_extractor.requires_grad_(False).train()
        acoustic.reconstruction_loss.train()
        acoustic.jdc_transform.train()
        entries.extend(
            (acoustic, name, getattr(acoustic, name))
            for name in (
                "audio_encoder",
                "feature_linear",
                "decoder",
                "generator",
                "discriminators",
            )
        )
    if conditional is not None:
        conditional.to(accelerator.device).train()
        conditional.text_encoder.requires_grad_(False).eval()
        if acoustic is None:
            conditional.audio_encoder.requires_grad_(False).train()
            conditional.f0_extractor.requires_grad_(False).train()
            conditional.decoder.requires_grad_(False).eval()
            conditional.generator.requires_grad_(False).eval()
        named = (
            "phoneme_encoder",
            "latent_phoneme_encoder",
            "duration_phoneme_encoder",
            "context_phoneme_encoder",
            "context_audio_encoder",
            "style_encoder",
            "voice_encoder",
            "language_embedding",
            "condition_bank",
            "duration_predictor",
            "latent_flow",
            "aligner",
            "style_speaker_classifier",
            "style_statistics_head",
            "voice_ge2e",
            "style_ge2e",
        )
        entries.extend(
            (conditional, name, getattr(conditional, name))
            for name in named
        )
    arguments: list[object] = [entry[2] for entry in entries]
    arguments.append(optimizers.generator)
    if optimizers.discriminator is not None:
        arguments.append(optimizers.discriminator)
    prepared = accelerator.prepare(*arguments)
    module_count = len(entries)
    for entry, module in zip(entries, prepared[:module_count], strict=True):
        setattr(entry[0], entry[1], module)
    optimizers.generator = prepared[module_count]
    if optimizers.discriminator is not None:
        optimizers.discriminator = prepared[module_count + 1]
    if conditional is not None and acoustic is not None:
        conditional.audio_encoder = acoustic.audio_encoder
        conditional.f0_extractor = acoustic.f0_extractor
        conditional.decoder = acoustic.decoder
        conditional.generator = acoustic.generator
    if models.latent_flow_ema is not None:
        models.latent_flow_ema.to(accelerator.device).requires_grad_(False).eval()
    return models, optimizers
