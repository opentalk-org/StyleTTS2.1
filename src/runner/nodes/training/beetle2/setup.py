from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator
from torch import nn

from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session

from .config import BeetleConfig, OptimizerConfig, TrainingStage
from .models import (
    AcousticModels,
    ConditionalDependencies,
    ConditionalModels,
    F0Extractor,
    FeatureLinear,
    build_acoustic_models,
    build_conditional_models,
    compile_acoustic,
)
from .models.modules.aligner import PhonemeAligner
from .models.modules.alignment_backbone import StyleTTSAlignerBackbone
from .models.modules.audio import AudioEncoder
from .models.modules.decoder import Decoder
from .models.modules.generator import Generator
from .models.modules.plbert import PlBertEncoder
from .phonemes import (
    LEGACY_SYMBOLS,
    PhonemeTokenizer,
    PlBertVocabulary,
    TextTokenizer,
)


@dataclass(frozen=True)
class TextResources:
    phoneme_tokenizer: PhonemeTokenizer
    text_tokenizer: TextTokenizer
    vocabulary: PlBertVocabulary
    plbert: PlBertEncoder | None


@dataclass
class TrainingOptimizers:
    generator: torch.optim.AdamW
    discriminator: torch.optim.AdamW | None


class FrozenAcoustic(nn.Module):
    def __init__(
        self,
        audio_encoder: AudioEncoder,
        feature_linear: FeatureLinear,
        f0_extractor: F0Extractor,
        decoder: Decoder,
        generator: Generator,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.feature_linear = feature_linear
        self.f0_extractor = f0_extractor
        self.decoder = decoder
        self.generator = generator


@dataclass
class TrainingModels:
    acoustic: AcousticModels | None
    conditional: ConditionalModels | None


def load_text_resources(config: BeetleConfig) -> TextResources:
    stage = config.training.stage
    root = Path(config.architecture.phoneme.model_path)
    vocabulary = PlBertVocabulary(root / "tokenizer")
    if len(vocabulary.symbols) != config.architecture.phoneme_token_count:
        raise ValueError(
            "PL-BERT vocabulary size does not match architecture.phoneme_token_count"
        )
    plbert = None
    if stage is not TrainingStage.POSTERIOR:
        plbert = PlBertEncoder(
            root,
            config.architecture.language.values,
            vocabulary.symbols,
        )
    return TextResources(
        PhonemeTokenizer(vocabulary),
        TextTokenizer(vocabulary),
        vocabulary,
        plbert,
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


def load_aligner(
    config: BeetleConfig,
    vocabulary: PlBertVocabulary,
) -> PhonemeAligner:
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
    aligner.load_checkpoint(checkpoint, LEGACY_SYMBOLS, vocabulary.symbols)
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
        assert resources.plbert is not None
        acoustic_dependency: nn.Module
        if acoustic is None:
            acoustic_dependency = FrozenAcoustic(
                AudioEncoder(config.architecture.posterior),
                FeatureLinear(config.architecture.feature),
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
                resources.plbert,
                load_aligner(config, resources.vocabulary),
            ),
        )
    if config.runtime.compile and acoustic is not None:
        compile_acoustic(acoustic)
    return TrainingModels(acoustic, conditional)


def trainable_conditional_modules(
    models: ConditionalModels,
) -> tuple[nn.Module, ...]:
    return (
        models.plbert,
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
    plbert_parameters: tuple[nn.Parameter, ...] = ()
    if models.conditional is not None:
        generator_modules.extend(trainable_conditional_modules(models.conditional))
        flow_parameters = tuple(models.conditional.latent_flow.parameters())
        plbert_parameters = tuple(models.conditional.plbert.parameters())
    parameters = tuple(
        parameter
        for module in generator_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    flow_ids = {id(parameter) for parameter in flow_parameters}
    plbert_ids = {id(parameter) for parameter in plbert_parameters}
    regular = tuple(
        parameter
        for parameter in parameters
        if id(parameter) not in flow_ids and id(parameter) not in plbert_ids
    )
    settings = config.training.generator_optimizer
    groups: list[dict[str, object]] = [
        {
            "params": regular,
            "weight_decay": settings.weight_decay,
            "learning_rate_scale": 1.0,
        }
    ]
    if flow_parameters:
        groups.append(
            {
                "params": flow_parameters,
                "weight_decay": config.training.latent_flow_weight_decay,
                "learning_rate_scale": 1.0,
            }
        )
    if plbert_parameters:
        groups.append(
            {
                "params": plbert_parameters,
                "weight_decay": config.architecture.phoneme.weight_decay,
                "learning_rate_scale": (
                    config.architecture.phoneme.learning_rate
                    / settings.learning_rate
                ),
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
        [{"params": parameters, "learning_rate_scale": 1.0}],
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
        if acoustic is None:
            conditional.audio_encoder.requires_grad_(False).train()
            conditional.feature_linear.requires_grad_(False).eval()
            conditional.f0_extractor.requires_grad_(False).train()
            conditional.decoder.requires_grad_(False).eval()
            conditional.generator.requires_grad_(False).eval()
        named = (
            "plbert",
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
        conditional.feature_linear = acoustic.feature_linear
        conditional.f0_extractor = acoustic.f0_extractor
        conditional.decoder = acoustic.decoder
        conditional.generator = acoustic.generator
    return models, optimizers
