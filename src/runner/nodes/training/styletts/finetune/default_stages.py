from .stages import (
    ProsodySource,
    StyleSource,
    TrainableModule,
    TrainingLoss,
    TrainingLossWeights,
    TrainingStageSpec,
    ValidationDurationSource,
    ValidationStageSpec,
)


def _validation(
    predicted_prosody: bool,
    alpha_flow: bool,
    predicted_duration: bool = False,
) -> ValidationStageSpec:
    prosody_source = (
        ProsodySource.PREDICTED
        if predicted_prosody
        else ProsodySource.GROUND_TRUTH
    )
    return ValidationStageSpec(
        f0_source=prosody_source,
        norm_source=prosody_source,
        duration_source=(
            ValidationDurationSource.PREDICTED
            if predicted_duration
            else ValidationDurationSource.GROUND_TRUTH
        ),
        alpha_flow=alpha_flow,
    )


def _loss_weights() -> TrainingLossWeights:
    return TrainingLossWeights(
        **{loss.value: 0 for loss in TrainingLoss}
    )


def build_default_training_stages() -> list[TrainingStageSpec]:
    weights = _loss_weights()
    mel_modules = [
        TrainableModule.TEXT_ENCODER,
        TrainableModule.VOICE_ENCODER,
        TrainableModule.DECODER,
        TrainableModule.TEXT_ALIGNER,
    ]
    tma_modules = list(mel_modules)
    continuous_prosody_modules = [
        TrainableModule.BERT,
        TrainableModule.BERT_ENCODER,
        TrainableModule.DURATION_PREDICTOR,
        TrainableModule.POSITION_EMBEDDING,
        TrainableModule.PROSODY_ENCODER,
        TrainableModule.PROSODY_PREDICTOR,
    ]
    quantized_prosody_modules = [
        TrainableModule.BERT,
        TrainableModule.BERT_ENCODER,
        TrainableModule.DURATION_PREDICTOR,
        TrainableModule.POSITION_EMBEDDING,
        TrainableModule.PROSODY_ENCODER,
        TrainableModule.PROSODY_PREDICTOR,
        TrainableModule.QUANTIZER,
    ]
    mel_weights = weights.model_copy(
        update={
            "mel": 5,
            "sequence_alignment": 0.2,
            "monotonic_alignment": 5,
        }
    )
    tma_weights = weights.model_copy(
        update={
            "mel": 5,
            "sequence_alignment": 0.2,
            "monotonic_alignment": 5,
            "adversarial": 1,
            "wavlm": 1,
            "speaker_feature": 1,
        }
    )
    prosody_weights = weights.model_copy(
        update={
            "f0": 1,
            "norm": 1,
            "duration": 1,
            "duration_ce": 20,
            "prosody_adversarial": 1,
        }
    )
    continuous_prosody_weights = prosody_weights.model_copy(
        update={
            "prosody_adversarial": 0.1,
        }
    )
    return [
        TrainingStageSpec(
            name="StyleTTS2 train_first.py · mel pretraining",
            steps=10_000,
            max_audio_seconds=150,
            max_decoder_seconds=9.0,
            voice_conditioning_dropout=0.2,
            style_source=StyleSource.CONTINUOUS,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=mel_modules,
            loss_weights=mel_weights,
            validation=_validation(False, False),
        ),
        TrainingStageSpec(
            name="StyleTTS2 train_first.py · TMA acoustic GAN",
            steps=10_000,
            max_audio_seconds=90,
            max_decoder_seconds=6.0,
            voice_conditioning_dropout=0.2,
            style_source=StyleSource.CONTINUOUS,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=tma_modules,
            loss_weights=tma_weights,
            validation=_validation(False, False),
        ),
        TrainingStageSpec(
            name="train_prosody_no_rvq.py · continuous · GAN x0.1",
            steps=5_000,
            max_audio_seconds=90,
            max_decoder_seconds=3.0,
            style_source=StyleSource.CONTINUOUS,
            prosody_source=ProsodySource.PREDICTED,
            trainable_modules=continuous_prosody_modules,
            loss_weights=continuous_prosody_weights,
            validation=_validation(True, False, predicted_duration=True),
        ),
        TrainingStageSpec(
            name="train_prosody.py · StyleTTS-ZS RVQ · GAN",
            steps=8_000,
            max_audio_seconds=75,
            max_decoder_seconds=3.0,
            style_source=StyleSource.QUANTIZED,
            prosody_source=ProsodySource.PREDICTED,
            trainable_modules=quantized_prosody_modules,
            loss_weights=prosody_weights,
            validation=_validation(True, False, predicted_duration=True),
        ),
        TrainingStageSpec(
            name="v_diffusion.py · AlphaFlow",
            steps=10_000,
            max_audio_seconds=65,
            max_decoder_seconds=3.0,
            style_source=StyleSource.QUANTIZED,
            prosody_source=ProsodySource.PREDICTED,
            trainable_modules=[TrainableModule.ALPHA_FLOW],
            loss_weights=weights.model_copy(update={"alpha_flow": 1}),
            validation=_validation(True, True, predicted_duration=True),
        ),
    ]
