from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProsodySource(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTED = "predicted"


class ReconstructionTarget(str, Enum):
    REAL_AUDIO = "real_audio"
    TEACHER_RECONSTRUCTION = "teacher_reconstruction"


class ValidationDurationSource(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTED = "predicted"


class ValidationStageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    f0_source: ProsodySource
    norm_source: ProsodySource
    duration_source: ValidationDurationSource
    diffusion: bool


class TrainableModule(str, Enum):
    BERT = "bert"
    BERT_ENCODER = "bert_encoder"
    DECODER = "decoder"
    DIFFUSION = "diffusion"
    PITCH_EXTRACTOR = "pitch_extractor"
    PREDICTOR = "predictor"
    PREDICTOR_ENCODER = "predictor_encoder"
    STYLE_ENCODER = "style_encoder"
    TEXT_ALIGNER = "text_aligner"
    TEXT_ENCODER = "text_encoder"


class TrainingLoss(str, Enum):
    ADVERSARIAL = "adversarial"
    DIFFUSION = "diffusion"
    DURATION = "duration"
    DURATION_CE = "duration_ce"
    F0 = "f0"
    MEL = "mel"
    MONOTONIC_ALIGNMENT = "monotonic_alignment"
    NORM = "norm"
    SEQUENCE_ALIGNMENT = "sequence_alignment"
    SLM_ADVERSARIAL = "slm_adversarial"
    STYLE = "style"
    WAVLM = "wavlm"


class TrainingLossWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adversarial: float = Field(ge=0)
    diffusion: float = Field(ge=0)
    duration: float = Field(ge=0)
    duration_ce: float = Field(ge=0)
    f0: float = Field(ge=0)
    mel: float = Field(ge=0)
    monotonic_alignment: float = Field(ge=0)
    norm: float = Field(ge=0)
    sequence_alignment: float = Field(ge=0)
    slm_adversarial: float = Field(ge=0)
    style: float = Field(ge=0)
    wavlm: float = Field(ge=0)


class TrainingStageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    steps: int = Field(gt=0)
    prosody_source: ProsodySource
    reconstruction_target: ReconstructionTarget
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
        if TrainingLoss.MEL not in self.enabled_losses:
            raise ValueError("every training stage must enable mel loss")
        if self.train_discriminators != (
            TrainingLoss.ADVERSARIAL in self.enabled_losses
        ):
            raise ValueError(
                "train_discriminators must match adversarial loss"
            )
        if (
            self.prosody_source is ProsodySource.GROUND_TRUTH
            and any(
                loss in self.enabled_losses
                for loss in (
                    TrainingLoss.F0,
                    TrainingLoss.NORM,
                    TrainingLoss.DURATION,
                    TrainingLoss.DURATION_CE,
                )
            )
        ):
            raise ValueError(
                "predicted prosody losses require prosody_source=predicted"
            )
        if (
            self.reconstruction_target
            is ReconstructionTarget.TEACHER_RECONSTRUCTION
            and self.prosody_source is ProsodySource.GROUND_TRUTH
        ):
            raise ValueError(
                "teacher reconstruction requires predicted prosody"
            )
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
    loss_weights = TrainingLossWeights(
        adversarial=1,
        diffusion=1,
        duration=1,
        duration_ce=20,
        f0=1,
        mel=5,
        monotonic_alignment=1,
        norm=1,
        sequence_alignment=1,
        slm_adversarial=1,
        style=1,
        wavlm=1,
    )
    shared_modules = [
        TrainableModule.BERT_ENCODER,
        TrainableModule.BERT,
        TrainableModule.PREDICTOR,
        TrainableModule.PREDICTOR_ENCODER,
        TrainableModule.STYLE_ENCODER,
        TrainableModule.DECODER,
        TrainableModule.TEXT_ENCODER,
        TrainableModule.TEXT_ALIGNER,
    ]
    shared_losses = [
        TrainingLoss.MEL,
        TrainingLoss.F0,
        TrainingLoss.NORM,
        TrainingLoss.DURATION,
        TrainingLoss.DURATION_CE,
        TrainingLoss.SEQUENCE_ALIGNMENT,
        TrainingLoss.MONOTONIC_ALIGNMENT,
        TrainingLoss.ADVERSARIAL,
        TrainingLoss.WAVLM,
    ]
    return [
        TrainingStageSpec(
            name="Finetune base",
            steps=100_000,
            prosody_source=ProsodySource.PREDICTED,
            reconstruction_target=ReconstructionTarget.REAL_AUDIO,
            trainable_modules=shared_modules,
            enabled_losses=shared_losses,
            loss_weights=loss_weights,
            train_discriminators=True,
            validation=ValidationStageSpec(
                f0_source=ProsodySource.PREDICTED,
                norm_source=ProsodySource.PREDICTED,
                duration_source=ValidationDurationSource.GROUND_TRUTH,
                diffusion=False,
            ),
        ),
        TrainingStageSpec(
            name="Finetune diffusion",
            steps=50_000,
            prosody_source=ProsodySource.PREDICTED,
            reconstruction_target=ReconstructionTarget.REAL_AUDIO,
            trainable_modules=[*shared_modules, TrainableModule.DIFFUSION],
            enabled_losses=[
                *shared_losses,
                TrainingLoss.STYLE,
                TrainingLoss.DIFFUSION,
            ],
            loss_weights=loss_weights,
            train_discriminators=True,
            validation=ValidationStageSpec(
                f0_source=ProsodySource.PREDICTED,
                norm_source=ProsodySource.PREDICTED,
                duration_source=ValidationDurationSource.GROUND_TRUTH,
                diffusion=False,
            ),
        ),
        TrainingStageSpec(
            name="Finetune joint",
            steps=25_000,
            prosody_source=ProsodySource.PREDICTED,
            reconstruction_target=ReconstructionTarget.REAL_AUDIO,
            trainable_modules=[*shared_modules, TrainableModule.DIFFUSION],
            enabled_losses=[
                *shared_losses,
                TrainingLoss.STYLE,
                TrainingLoss.DIFFUSION,
                TrainingLoss.SLM_ADVERSARIAL,
            ],
            loss_weights=loss_weights,
            train_discriminators=True,
            validation=ValidationStageSpec(
                f0_source=ProsodySource.PREDICTED,
                norm_source=ProsodySource.PREDICTED,
                duration_source=ValidationDurationSource.GROUND_TRUTH,
                diffusion=False,
            ),
        ),
    ]
