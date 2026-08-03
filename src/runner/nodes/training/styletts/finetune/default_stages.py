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
) -> ValidationStageSpec:
    prosody_source = (
        ProsodySource.PREDICTED
        if predicted_prosody
        else ProsodySource.GROUND_TRUTH
    )
    return ValidationStageSpec(
        f0_source=prosody_source,
        norm_source=prosody_source,
        duration_source=ValidationDurationSource.GROUND_TRUTH,
        alpha_flow=alpha_flow,
    )


def _loss_weights() -> TrainingLossWeights:
    return TrainingLossWeights(
        **{loss.value: 0 for loss in TrainingLoss}
    )


def build_default_training_stages() -> list[TrainingStageSpec]:
    weights = _loss_weights()
    acoustic_modules = [
        TrainableModule.TEXT_ENCODER,
        TrainableModule.VOICE_ENCODER,
        TrainableModule.DECODER,
        TrainableModule.TEXT_ALIGNER,
    ]
    prosody_modules = [
        TrainableModule.DURATION_PREDICTOR,
        TrainableModule.PROSODY_ENCODER,
        TrainableModule.PROSODY_PREDICTOR,
        TrainableModule.QUANTIZER,
        TrainableModule.POSITION_EMBEDDING,
        TrainableModule.FACTORIZATION,
    ]
    acoustic_weights = weights.model_copy(
        update={
            "mel": 5,
            "sequence_alignment": 1,
            "monotonic_alignment": 10,
            "adversarial": 1,
            "wavlm": 1,
            "slm_adversarial": 1,
            "speaker_feature": 5,
            "speaker_similarity": 5,
        }
    )
    prosody_weights = weights.model_copy(
        update={
            "f0": 1,
            "norm": 1,
            "duration": 1,
            "duration_ce": 20,
            "prosody_adversarial": 1,
            "rvq": 1,
            "style_nuisance": 0.1,
            "xcov": 0.01,
        }
    )
    return [
        TrainingStageSpec(
            name="train_test.py · acoustic training",
            steps=2_000,
            batch_size=10,
            style_source=StyleSource.CONTINUOUS,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=acoustic_modules,
            loss_weights=acoustic_weights,
            max_decoder_seconds=3.75,
            validation=_validation(False, False),
        ),
        TrainingStageSpec(
            name="small_rec.py · prosody/RVQ and factorization",
            steps=2_000,
            batch_size=10,
            style_source=StyleSource.QUANTIZED,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=prosody_modules,
            loss_weights=prosody_weights,
            validation=_validation(True, False),
        ),
        TrainingStageSpec(
            name="v_diffusion.py · AlphaFlow",
            steps=2_000,
            batch_size=10,
            style_source=StyleSource.QUANTIZED,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=[TrainableModule.ALPHA_FLOW],
            loss_weights=weights.model_copy(update={"alpha_flow": 1}),
            validation=_validation(True, True),
        ),
    ]
