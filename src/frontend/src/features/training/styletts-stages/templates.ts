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
  "sequence_alignment",
  "slm_adversarial",
  "speaker_feature",
  "speaker_similarity",
  "style_nuisance",
  "wavlm",
  "xcov",
] as const;

export type TrainableModule = (typeof TRAINABLE_MODULES)[number];
export type TrainingLoss = (typeof TRAINING_LOSSES)[number];
export type TrainingLossWeights = Record<TrainingLoss, number>;

export const TRAINING_LOSS_LABELS: Record<TrainingLoss, string> = {
  alpha_flow: "AlphaFlow",
  adversarial: "Waveform adversarial",
  duration: "Duration",
  duration_ce: "Duration cross-entropy",
  f0: "F0 reconstruction",
  mel: "Mel reconstruction",
  monotonic_alignment: "Monotonic alignment",
  norm: "Energy / norm reconstruction",
  prosody_adversarial: "Prosody GAN + feature matching",
  sequence_alignment: "Sequence alignment",
  slm_adversarial: "SLM adversarial",
  speaker_feature: "Speaker feature",
  speaker_similarity: "Speaker similarity",
  style_nuisance: "Style nuisance",
  wavlm: "WavLM reconstruction",
  xcov: "Cross-covariance",
};

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
  { label: "TMA GAN warm-up", indexes: [1] },
  { label: "TMA acoustic GAN", indexes: [1, 2] },
  { label: "Prosody without GAN", indexes: [3] },
  { label: "Prosody GAN", indexes: [4] },
  { label: "AlphaFlow", indexes: [5] },
  { label: "StyleTTS2 acoustic", indexes: [0, 1, 2] },
  { label: "Default StyleTTS-ZS", indexes: [0, 1, 2, 3, 4, 5] },
  { label: "Without mel pretraining", indexes: [1, 2, 3, 4, 5] },
] as const;

export function stageTemplates(defaultStages: TrainingStageSpec[]): TrainingStageSpec[] {
  return defaultStages.map(cloneStage);
}

export function cloneStage(stage: TrainingStageSpec): TrainingStageSpec {
  return structuredClone(stage);
}
