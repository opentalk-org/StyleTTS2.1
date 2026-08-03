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
  "language_embedding",
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
  batch_size: number;
  max_audio_seconds: number;
  max_decoder_seconds: number;
  style_source: StyleSource;
  prosody_source: ProsodySource;
  trainable_modules: TrainableModule[];
  loss_weights: TrainingLossWeights;
  validation: ValidationStageSpec;
};

export const DEFAULT_LOSS_WEIGHTS: TrainingLossWeights = {
  alpha_flow: 0,
  adversarial: 0,
  duration: 0,
  duration_ce: 0,
  f0: 0,
  mel: 0,
  monotonic_alignment: 0,
  norm: 0,
  prosody_adversarial: 0,
  rvq: 0,
  sequence_alignment: 0,
  slm_adversarial: 0,
  speaker_feature: 0,
  speaker_similarity: 0,
  wavlm: 0,
  style_nuisance: 0,
  xcov: 0,
};

export const DEFAULT_VALIDATION: ValidationStageSpec = {
  f0_source: "ground_truth",
  norm_source: "ground_truth",
  duration_source: "ground_truth",
  alpha_flow: false,
};

const PREDICTED_PROSODY_VALIDATION: ValidationStageSpec = {
  ...DEFAULT_VALIDATION,
  f0_source: "predicted",
  norm_source: "predicted",
};

const ACOUSTIC_MODULES: TrainableModule[] = [
  "text_encoder",
  "voice_encoder",
  "decoder",
  "text_aligner",
];
const PROSODY_MODULES: TrainableModule[] = [
  "duration_predictor",
  "prosody_encoder",
  "prosody_predictor",
  "quantizer",
  "position_embedding",
  "factorization",
];
export const STAGE_TEMPLATES: TrainingStageSpec[] = [
  {
    name: "train_test.py · acoustic training",
    steps: 2_000,
    batch_size: 28,
    max_audio_seconds: 15,
    max_decoder_seconds: 3.5,
    style_source: "continuous",
    prosody_source: "ground_truth",
    trainable_modules: ACOUSTIC_MODULES,
    loss_weights: {
      ...DEFAULT_LOSS_WEIGHTS,
      mel: 5,
      sequence_alignment: 1,
      monotonic_alignment: 1,
      adversarial: 1,
      wavlm: 1,
      slm_adversarial: 1,
      speaker_feature: 5,
      speaker_similarity: 5,
    },
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "small_rec.py · prosody/RVQ and factorization",
    steps: 2_000,
    batch_size: 28,
    max_audio_seconds: 15,
    max_decoder_seconds: 3.5,
    style_source: "quantized",
    prosody_source: "ground_truth",
    trainable_modules: PROSODY_MODULES,
    loss_weights: {
      ...DEFAULT_LOSS_WEIGHTS,
      f0: 1,
      norm: 1,
      duration: 1,
      duration_ce: 20,
      prosody_adversarial: 1,
      rvq: 1,
      style_nuisance: 0.1,
      xcov: 0.01,
    },
    validation: { ...PREDICTED_PROSODY_VALIDATION },
  },
  {
    name: "v_diffusion.py · AlphaFlow",
    steps: 2_000,
    batch_size: 28,
    max_audio_seconds: 15,
    max_decoder_seconds: 3.5,
    style_source: "quantized",
    prosody_source: "ground_truth",
    trainable_modules: ["alpha_flow"],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS, alpha_flow: 1 },
    validation: { ...PREDICTED_PROSODY_VALIDATION, alpha_flow: true },
  },
];

export const STAGE_PRESETS = [
  { label: "train_test.py", indexes: [0] },
  { label: "small_rec.py", indexes: [1] },
  { label: "v_diffusion.py → AlphaFlow", indexes: [2] },
  { label: "Full StyleTTS-ZS", indexes: [0, 1, 2] },
] as const;

export function cloneStage(stage: TrainingStageSpec): TrainingStageSpec {
  return structuredClone(stage);
}
