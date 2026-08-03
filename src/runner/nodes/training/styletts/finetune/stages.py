from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProsodySource(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTED = "predicted"


class StyleSource(str, Enum):
    CONTINUOUS = "continuous"
    QUANTIZED = "quantized"


class ValidationDurationSource(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTED = "predicted"


class ValidationStageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    f0_source: ProsodySource
    norm_source: ProsodySource
    duration_source: ValidationDurationSource
    alpha_flow: bool


class TrainableModule(str, Enum):
    ALPHA_FLOW = "alpha_flow"
    BERT = "bert"
    BERT_ENCODER = "bert_encoder"
    DECODER = "decoder"
    DURATION_PREDICTOR = "duration_predictor"
    FACTORIZATION = "factorization"
    POSITION_EMBEDDING = "position_embedding"
    PROSODY_ENCODER = "prosody_encoder"
    PROSODY_PREDICTOR = "prosody_predictor"
    QUANTIZER = "quantizer"
    TEXT_ALIGNER = "text_aligner"
    TEXT_ENCODER = "text_encoder"
    VOICE_ENCODER = "voice_encoder"


class TrainingLoss(str, Enum):
    ALPHA_FLOW = "alpha_flow"
    ADVERSARIAL = "adversarial"
    DURATION = "duration"
    DURATION_CE = "duration_ce"
    F0 = "f0"
    MEL = "mel"
    MONOTONIC_ALIGNMENT = "monotonic_alignment"
    NORM = "norm"
    PROSODY_ADVERSARIAL = "prosody_adversarial"
    RVQ = "rvq"
    SEQUENCE_ALIGNMENT = "sequence_alignment"
    SLM_ADVERSARIAL = "slm_adversarial"
    SPEAKER_FEATURE = "speaker_feature"
    SPEAKER_SIMILARITY = "speaker_similarity"
    WAVLM = "wavlm"
    STYLE_NUISANCE = "style_nuisance"
    XCOV = "xcov"


class TrainingLossWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha_flow: float = Field(ge=0)
    adversarial: float = Field(ge=0)
    duration: float = Field(ge=0)
    duration_ce: float = Field(ge=0)
    f0: float = Field(ge=0)
    mel: float = Field(ge=0)
    monotonic_alignment: float = Field(ge=0)
    norm: float = Field(ge=0)
    prosody_adversarial: float = Field(default=1, ge=0)
    rvq: float = Field(ge=0)
    sequence_alignment: float = Field(ge=0)
    slm_adversarial: float = Field(ge=0)
    speaker_feature: float = Field(ge=0)
    speaker_similarity: float = Field(ge=0)
    wavlm: float = Field(ge=0)
    style_nuisance: float = Field(ge=0)
    xcov: float = Field(ge=0)


class TrainingStageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    steps: int = Field(gt=0)
    batch_size: int = Field(default=28, ge=1, le=128)
    max_audio_seconds: float = Field(default=15.0, ge=1, le=60)
    max_decoder_seconds: float = Field(default=3.5, ge=1, le=30)
    style_source: StyleSource = StyleSource.QUANTIZED
    prosody_source: ProsodySource
    trainable_modules: list[TrainableModule]
    enabled_losses: list[TrainingLoss]
    loss_weights: TrainingLossWeights
    train_discriminators: bool
    validation: ValidationStageSpec

    @model_validator(mode="after")
    def validate_policy(self) -> "TrainingStageSpec":
        if len(set(self.trainable_modules)) != len(self.trainable_modules):
            raise ValueError("trainable_modules must not contain duplicates")
        if len(set(self.enabled_losses)) != len(self.enabled_losses):
            raise ValueError("enabled_losses must not contain duplicates")
        if self.train_discriminators != (
            TrainingLoss.ADVERSARIAL in self.enabled_losses
        ):
            raise ValueError(
                "train_discriminators must match adversarial loss"
            )
        if (
            TrainingLoss.ADVERSARIAL in self.enabled_losses
            and TrainingLoss.MEL not in self.enabled_losses
        ):
            raise ValueError("waveform adversarial training requires mel reconstruction")
        return self


def stage_for_step(
    stages: list[TrainingStageSpec],
    step: int,
) -> TrainingStageSpec:
    boundary = 0
    for stage in stages:
        boundary += stage.steps
        if step < boundary:
            return stage
    raise ValueError(f"training step {step} exceeds configured stage schedule")


def default_training_stages() -> list[TrainingStageSpec]:
    # Local import breaks the intentional cycle: defaults are built from the
    # validated stage types defined in this module.
    from .default_stages import build_default_training_stages

    return build_default_training_stages()
