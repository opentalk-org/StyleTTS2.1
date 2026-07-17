from dataclasses import dataclass

from torch import nn
from transformers import BertModel

from ..config import BeetleConfig
from ..losses.embeddings import GE2ELoss
from .modules.aligner import PhonemeAligner
from .modules.conditioning import ConditionBank, ConditionChannels
from .modules.duration import DurationPredictor
from .modules.embeddings import (
    StyleEncoder,
    StyleSpeakerClassifier,
    StyleStatisticsHead,
    VoiceEncoder,
)
from .modules.latent_flow import LatentFlowModel
from .modules.text import (
    ContextAudioEncoder,
    ContextPhonemeEncoder,
    DurationPhonemeEncoder,
    LatentPhonemeEncoder,
    PhonemeEncoder,
    TextEncoder,
)


@dataclass(frozen=True)
class Stage2Dependencies:
    phoneme_bert: BertModel
    text_bert: BertModel
    aligner: PhonemeAligner


@dataclass(frozen=True)
class Stage2ParameterReport:
    inference: int
    frozen_helper: int
    training_only: int
    excluded_text: int

    @property
    def total(self) -> int:
        return (
            self.inference
            + self.frozen_helper
            + self.training_only
            + self.excluded_text
        )


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


class Stage2Models(nn.Module):
    def __init__(
        self,
        audio_encoder: nn.Module,
        f0_extractor: nn.Module,
        phoneme_encoder: PhonemeEncoder,
        latent_phoneme_encoder: LatentPhonemeEncoder,
        duration_phoneme_encoder: DurationPhonemeEncoder,
        context_phoneme_encoder: ContextPhonemeEncoder,
        context_audio_encoder: ContextAudioEncoder,
        style_encoder: StyleEncoder,
        voice_encoder: VoiceEncoder,
        condition_bank: ConditionBank,
        duration_predictor: DurationPredictor,
        latent_flow: LatentFlowModel,
        aligner: PhonemeAligner,
        text_encoder: TextEncoder,
        style_speaker_classifier: StyleSpeakerClassifier,
        style_statistics_head: StyleStatisticsHead,
        voice_ge2e: GE2ELoss,
        style_ge2e: GE2ELoss,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.f0_extractor = f0_extractor
        self.phoneme_encoder = phoneme_encoder
        self.latent_phoneme_encoder = latent_phoneme_encoder
        self.duration_phoneme_encoder = duration_phoneme_encoder
        self.context_phoneme_encoder = context_phoneme_encoder
        self.context_audio_encoder = context_audio_encoder
        self.style_encoder = style_encoder
        self.voice_encoder = voice_encoder
        self.condition_bank = condition_bank
        self.duration_predictor = duration_predictor
        self.latent_flow = latent_flow
        self.aligner = aligner
        self.text_encoder = text_encoder
        self.style_speaker_classifier = style_speaker_classifier
        self.style_statistics_head = style_statistics_head
        self.voice_ge2e = voice_ge2e
        self.style_ge2e = style_ge2e

    def parameter_report(
        self, stage1_inference_parameters: int
    ) -> Stage2ParameterReport:
        inference_modules = (
            self.phoneme_encoder,
            self.latent_phoneme_encoder,
            self.duration_phoneme_encoder,
            self.context_phoneme_encoder,
            self.context_audio_encoder,
            self.style_encoder,
            self.voice_encoder,
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
        inference, inference_ids = _parameter_count(inference_modules)
        helper, helper_ids = _parameter_count((self.f0_extractor,))
        training, training_ids = _parameter_count(training_modules)
        text, text_ids = _parameter_count((self.text_encoder,))
        categories = (inference_ids, helper_ids, training_ids, text_ids)
        for index, identities in enumerate(categories):
            if any(identities & other for other in categories[index + 1 :]):
                raise ValueError("Stage 2 parameter categories must be disjoint")
        return Stage2ParameterReport(
            inference=stage1_inference_parameters + inference,
            frozen_helper=helper,
            training_only=training,
            excluded_text=text,
        )


def build_stage2_models(
    config: BeetleConfig,
    stage1: nn.Module,
    dependencies: Stage2Dependencies,
) -> Stage2Models:
    architecture = config.architecture
    if (
        dependencies.text_bert.config.hidden_size
        != architecture.text_encoder.hidden_channels
    ):
        raise ValueError("text BERT hidden width does not match text configuration")
    stage1.requires_grad_(False)
    text_encoder = TextEncoder(
        dependencies.text_bert,
        architecture.text_encoder.projection_channels,
    ).requires_grad_(False)
    condition_channels = ConditionChannels(
        phoneme=architecture.phoneme.cnn_hidden_channels,
        style=architecture.embeddings.embedding_channels,
        voice=architecture.embeddings.embedding_channels,
        pooled_phoneme=architecture.phoneme.projection_channels,
        pre_text=architecture.phoneme.cnn_hidden_channels,
        post_text=architecture.phoneme.cnn_hidden_channels,
        pre_audio=architecture.context.output_channels,
        post_audio=architecture.context.output_channels,
    )
    return Stage2Models(
        audio_encoder=stage1.audio_encoder,
        f0_extractor=stage1.f0_extractor,
        phoneme_encoder=PhonemeEncoder(
            dependencies.phoneme_bert,
            architecture.phoneme.projection_channels,
        ),
        latent_phoneme_encoder=LatentPhonemeEncoder(architecture.phoneme),
        duration_phoneme_encoder=DurationPhonemeEncoder(architecture.phoneme),
        context_phoneme_encoder=ContextPhonemeEncoder(architecture.phoneme),
        context_audio_encoder=ContextAudioEncoder(
            architecture.posterior.latent_channels,
            architecture.context,
        ),
        style_encoder=StyleEncoder(architecture.embeddings),
        voice_encoder=VoiceEncoder(architecture.embeddings),
        condition_bank=ConditionBank(
            condition_channels,
            architecture.conditioning.common_channels,
        ),
        duration_predictor=DurationPredictor(architecture.duration_flow),
        latent_flow=LatentFlowModel(
            architecture.latent_flow,
            architecture.conditioning.concat_layers,
        ),
        aligner=dependencies.aligner,
        text_encoder=text_encoder,
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
