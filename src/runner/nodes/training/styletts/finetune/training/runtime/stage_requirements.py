from ...stages import TrainingLoss


VOICE_LOSSES = {
    TrainingLoss.ADVERSARIAL,
    TrainingLoss.MEL,
    TrainingLoss.SLM_ADVERSARIAL,
    TrainingLoss.SPEAKER_FEATURE,
    TrainingLoss.SPEAKER_SIMILARITY,
    TrainingLoss.STYLE_NUISANCE,
    TrainingLoss.WAVLM,
    TrainingLoss.XCOV,
}


def requires_voice(enabled: set[TrainingLoss]) -> bool:
    return bool(enabled & VOICE_LOSSES)
