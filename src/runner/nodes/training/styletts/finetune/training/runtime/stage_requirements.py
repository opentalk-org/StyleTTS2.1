from ...stages import TrainingLoss


VOICE_LOSSES = {
    TrainingLoss.ADVERSARIAL,
    TrainingLoss.MEL,
    TrainingLoss.SLM_ADVERSARIAL,
    TrainingLoss.STYLE_NUISANCE,
    TrainingLoss.VOICE_METRIC,
    TrainingLoss.VOICE_NUISANCE,
    TrainingLoss.VOICE_PAIR,
    TrainingLoss.WAVLM,
    TrainingLoss.XCOV,
}


def requires_voice(enabled: set[TrainingLoss]) -> bool:
    return bool(enabled & VOICE_LOSSES)


def requires_generated_style(enabled: set[TrainingLoss]) -> bool:
    return TrainingLoss.STYLE in enabled
