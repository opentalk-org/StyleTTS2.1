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
        alpha_flow=1,
        adversarial=1,
        duration=1,
        duration_ce=20,
        f0=1,
        mel=5,
        monotonic_alignment=1,
        norm=1,
        prosody_adversarial=1,
        rvq=1,
        sequence_alignment=1,
        slm_adversarial=1,
        speaker_feature=5,
        speaker_similarity=5,
        wavlm=1,
        style_nuisance=0.1,
        xcov=0.01,
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
    prosody_losses = [
        TrainingLoss.F0,
        TrainingLoss.NORM,
        TrainingLoss.DURATION,
        TrainingLoss.DURATION_CE,
        TrainingLoss.PROSODY_ADVERSARIAL,
        TrainingLoss.RVQ,
        TrainingLoss.STYLE_NUISANCE,
        TrainingLoss.XCOV,
    ]
    common = dict(
        loss_weights=weights,
        train_discriminators=False,
    )
    return [
        TrainingStageSpec(
            name="train_test.py · acoustic training",
            steps=100_000,
            style_source=StyleSource.CONTINUOUS,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=acoustic_modules,
            enabled_losses=[
                TrainingLoss.MEL,
                TrainingLoss.SEQUENCE_ALIGNMENT,
                TrainingLoss.MONOTONIC_ALIGNMENT,
                TrainingLoss.ADVERSARIAL,
                TrainingLoss.WAVLM,
                TrainingLoss.SLM_ADVERSARIAL,
                TrainingLoss.SPEAKER_FEATURE,
                TrainingLoss.SPEAKER_SIMILARITY,
            ],
            validation=_validation(False, False),
            train_discriminators=True,
            **{
                key: value
                for key, value in common.items()
                if key != "train_discriminators"
            },
        ),
        TrainingStageSpec(
            name="small_rec.py · prosody/RVQ and factorization",
            steps=100_000,
            style_source=StyleSource.QUANTIZED,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=prosody_modules,
            enabled_losses=prosody_losses,
            validation=_validation(True, False),
            **common,
        ),
        TrainingStageSpec(
            name="v_diffusion.py · AlphaFlow",
            steps=50_000,
            style_source=StyleSource.QUANTIZED,
            prosody_source=ProsodySource.GROUND_TRUTH,
            trainable_modules=[TrainableModule.ALPHA_FLOW],
            enabled_losses=[TrainingLoss.ALPHA_FLOW],
            validation=_validation(True, True),
            **common,
        ),
    ]
