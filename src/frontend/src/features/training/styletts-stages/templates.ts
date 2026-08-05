export type ProsodySource = "ground_truth" | "predicted";
export type StyleSource = "continuous" | "quantized";
export type ValidationDurationSource = "ground_truth" | "predicted";

export type ValidationStageSpec = {
  f0_source: ProsodySource;
  norm_source: ProsodySource;
  duration_source: ValidationDurationSource;
  alpha_flow: boolean;
};

export const TRAINABLE_MODULES = [
  "alpha_flow",
  "bert",
  "bert_encoder",
  "decoder",
  "duration_predictor",
  "factorization",
  "position_embedding",
  "prosody_encoder",
  "prosody_predictor",
  "quantizer",
  "text_aligner",
  "text_encoder",
  "voice_encoder",
] as const;

export const TRAINING_LOSSES = [
  "alpha_flow",
  "adversarial",
  "duration",
  "duration_ce",
  "f0",
  "mel",
  "monotonic_alignment",
  "norm",
  "prosody_adversarial",
  "rvq",
  "sequence_alignment",
  "slm_adversarial",
  "speaker_feature",
  "speaker_similarity",
  "wavlm",
  "style_nuisance",
  "xcov",
] as const;

export type TrainableModule = (typeof TRAINABLE_MODULES)[number];
export type TrainingLoss = (typeof TRAINING_LOSSES)[number];
export type TrainingLossWeights = Record<TrainingLoss, number>;

export type TrainingStageSpec = {
  name: string;
  steps: number;
  max_audio_seconds: number;
  max_decoder_seconds: number;
  voice_conditioning_dropout: number;
  style_source: StyleSource;
  prosody_source: ProsodySource;
  trainable_modules: TrainableModule[];
  loss_weights: TrainingLossWeights;
  validation: ValidationStageSpec;
};

export const STAGE_PRESETS = [
  { label: "Mel pretraining", indexes: [0] },
  { label: "TMA acoustic training", indexes: [1] },
  { label: "Prosody / RVQ", indexes: [2] },
  { label: "AlphaFlow", indexes: [3] },
  { label: "StyleTTS2 acoustic", indexes: [0, 1] },
  { label: "Full StyleTTS-ZS", indexes: [0, 1, 2, 3] },
] as const;

export function cloneStage(stage: TrainingStageSpec): TrainingStageSpec {
  return structuredClone(stage);
}
