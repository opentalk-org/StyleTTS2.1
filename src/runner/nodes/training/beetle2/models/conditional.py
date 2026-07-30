from dataclasses import dataclass

from torch import nn

from ..config import BeetleConfig
from ..losses.embeddings import GE2ELoss
from .modules.aligner import PhonemeAligner
from .modules.conditioning import ConditionBank, ConditionChannels
from .modules.duration import DurationPredictor
from .modules.embeddings import (
    LanguageEmbedding,
    StyleEncoder,
    StyleSpeakerClassifier,
    StyleStatisticsHead,
    VoiceEncoder,
)
from .modules.latent_flow import LatentFlowModel
from .modules.plbert import PlBertEncoder
from .modules.text import (
    ContextAudioEncoder,
    ContextPhonemeEncoder,
    DurationPhonemeEncoder,
    LatentPhonemeEncoder,
    PhonemeEncoder,
)


@dataclass(frozen=True)
class ConditionalDependencies:
    plbert: PlBertEncoder
    aligner: PhonemeAligner


@dataclass(frozen=True)
class ConditionalParameterReport:
    inference: int
    frozen_helper: int
    training_only: int

    @property
    def total(self) -> int:
        return self.inference + self.frozen_helper + self.training_only


def _parameter_count(modules: tuple[nn.Module, ...]) -> tuple[int, set[int]]:
    identities: set[int] = set()
    count = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in identities:
                identities.add(identity)
                count += parameter.numel()
    return count, identities


class ConditionalModels(nn.Module):
    def __init__(
        self,
        audio_encoder: nn.Module,
        feature_linear: nn.Module,
        f0_extractor: nn.Module,
        decoder: nn.Module,
        generator: nn.Module,
        plbert: PlBertEncoder,
        phoneme_encoder: PhonemeEncoder,
        latent_phoneme_encoder: LatentPhonemeEncoder,
        duration_phoneme_encoder: DurationPhonemeEncoder,
        context_phoneme_encoder: ContextPhonemeEncoder,
        context_audio_encoder: ContextAudioEncoder,
        style_encoder: StyleEncoder,
        voice_encoder: VoiceEncoder,
        language_embedding: LanguageEmbedding,
        condition_bank: ConditionBank,
        duration_predictor: DurationPredictor,
        latent_flow: LatentFlowModel,
        aligner: PhonemeAligner,
        style_speaker_classifier: StyleSpeakerClassifier,
        style_statistics_head: StyleStatisticsHead,
        voice_ge2e: GE2ELoss,
        style_ge2e: GE2ELoss,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.feature_linear = feature_linear
        self.f0_extractor = f0_extractor
        self.decoder = decoder
        self.generator = generator
        self.plbert = plbert
        self.phoneme_encoder = phoneme_encoder
        self.latent_phoneme_encoder = latent_phoneme_encoder
        self.duration_phoneme_encoder = duration_phoneme_encoder
        self.context_phoneme_encoder = context_phoneme_encoder
        self.context_audio_encoder = context_audio_encoder
        self.style_encoder = style_encoder
        self.voice_encoder = voice_encoder
        self.language_embedding = language_embedding
        self.condition_bank = condition_bank
        self.duration_predictor = duration_predictor
        self.latent_flow = latent_flow
        self.aligner = aligner
        self.style_speaker_classifier = style_speaker_classifier
        self.style_statistics_head = style_statistics_head
        self.voice_ge2e = voice_ge2e
        self.style_ge2e = style_ge2e

    def parameter_report(
        self, acoustic_inference_parameters: int
    ) -> ConditionalParameterReport:
        inference_modules = (
            self.plbert,
            self.phoneme_encoder,
            self.latent_phoneme_encoder,
            self.duration_phoneme_encoder,
            self.context_phoneme_encoder,
            self.context_audio_encoder,
            self.style_encoder,
            self.voice_encoder,
            self.language_embedding,
            self.condition_bank,
            self.duration_predictor,
            self.latent_flow,
        )
        training_modules = (
            self.aligner,
            self.style_speaker_classifier,
            self.style_statistics_head,
            self.voice_ge2e,
            self.style_ge2e,
        )
        inference, _ = _parameter_count(inference_modules)
        helper, _ = _parameter_count((self.f0_extractor,))
        training, _ = _parameter_count(training_modules)
        return ConditionalParameterReport(
            inference=acoustic_inference_parameters + inference,
            frozen_helper=helper,
            training_only=training,
        )


def build_conditional_models(
    config: BeetleConfig,
    acoustic: nn.Module,
    dependencies: ConditionalDependencies,
) -> ConditionalModels:
    architecture = config.architecture
    condition_channels = ConditionChannels(
        phoneme=architecture.phoneme.cnn_hidden_channels,
        style=architecture.embeddings.embedding_channels,
        voice=architecture.embeddings.embedding_channels,
        pooled_phoneme=architecture.phoneme.projection_channels,
        pre_text=architecture.context.output_channels,
        post_text=architecture.context.output_channels,
        pre_audio=architecture.context.output_channels,
        post_audio=architecture.context.output_channels,
        language=architecture.language.embedding_channels,
    )
    return ConditionalModels(
        audio_encoder=acoustic.audio_encoder,
        feature_linear=acoustic.feature_linear,
        f0_extractor=acoustic.f0_extractor,
        decoder=acoustic.decoder,
        generator=acoustic.generator,
        plbert=dependencies.plbert,
        phoneme_encoder=PhonemeEncoder(
            dependencies.plbert.output_channels,
            architecture.phoneme.projection_channels,
        ),
        latent_phoneme_encoder=LatentPhonemeEncoder(
            architecture.phoneme,
            dependencies.plbert.maximum_positions,
        ),
        duration_phoneme_encoder=DurationPhonemeEncoder(architecture.phoneme),
        context_phoneme_encoder=ContextPhonemeEncoder(
            architecture.phoneme.projection_channels,
            architecture.context,
        ),
        context_audio_encoder=ContextAudioEncoder(
            architecture.posterior.latent_channels,
            architecture.context,
        ),
        style_encoder=StyleEncoder(architecture.embeddings),
        voice_encoder=VoiceEncoder(architecture.embeddings),
        language_embedding=LanguageEmbedding(
            len(architecture.language.values),
            architecture.language.embedding_channels,
        ),
        condition_bank=ConditionBank(
            condition_channels,
            architecture.conditioning.common_channels,
        ),
        duration_predictor=DurationPredictor(architecture.duration_flow),
        latent_flow=LatentFlowModel(architecture.latent_flow),
        aligner=dependencies.aligner,
        style_speaker_classifier=StyleSpeakerClassifier(
            architecture.embeddings.embedding_channels,
            architecture.embeddings.speaker_classes,
        ),
        style_statistics_head=StyleStatisticsHead(
            architecture.embeddings.embedding_channels
        ),
        voice_ge2e=GE2ELoss(initial_scale=10.0, initial_bias=-5.0),
        style_ge2e=GE2ELoss(initial_scale=10.0, initial_bias=-5.0),
    )
