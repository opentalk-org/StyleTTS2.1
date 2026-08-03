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
  "pitch_extractor",
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
  "wavlm",
  "style_nuisance",
  "voice_metric",
  "voice_nuisance",
  "voice_pair",
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
  enabled_losses: TrainingLoss[];
  loss_weights: TrainingLossWeights;
  train_discriminators: boolean;
  validation: ValidationStageSpec;
};

export const DEFAULT_LOSS_WEIGHTS: TrainingLossWeights = {
  alpha_flow: 1,
  adversarial: 1,
  duration: 1,
  duration_ce: 20,
  f0: 1,
  mel: 5,
  monotonic_alignment: 1,
  norm: 1,
  prosody_adversarial: 1,
  rvq: 1,
  sequence_alignment: 1,
  slm_adversarial: 1,
  wavlm: 1,
  style_nuisance: 0.1,
  voice_metric: 1,
  voice_nuisance: 0.1,
  voice_pair: 1,
  xcov: 0.01,
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
  "pitch_extractor",
  "factorization",
];
const PROSODY_MODULES: TrainableModule[] = [
  "duration_predictor",
  "prosody_encoder",
  "prosody_predictor",
  "quantizer",
  "position_embedding",
  "factorization",
];
const PROSODY_LOSSES: TrainingLoss[] = [
  "f0",
  "norm",
  "duration",
  "duration_ce",
  "prosody_adversarial",
  "rvq",
  "style_nuisance",
  "xcov",
];

export const STAGE_TEMPLATES: TrainingStageSpec[] = [
  {
    name: "train_test.py · acoustic training",
    steps: 100_000,
    batch_size: 28,
    max_audio_seconds: 15,
    max_decoder_seconds: 3,
    style_source: "continuous",
    prosody_source: "ground_truth",
    trainable_modules: ACOUSTIC_MODULES,
    enabled_losses: [
      "mel",
      "sequence_alignment",
      "monotonic_alignment",
      "adversarial",
      "wavlm",
      "slm_adversarial",
      "voice_pair",
      "voice_metric",
      "voice_nuisance",
    ],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: true,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "small_rec.py · prosody/RVQ and factorization",
    steps: 100_000,
    batch_size: 28,
    max_audio_seconds: 15,
    max_decoder_seconds: 3,
    style_source: "quantized",
    prosody_source: "ground_truth",
    trainable_modules: PROSODY_MODULES,
    enabled_losses: PROSODY_LOSSES,
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: false,
    validation: { ...PREDICTED_PROSODY_VALIDATION },
  },
  {
    name: "v_diffusion.py · AlphaFlow",
    steps: 50_000,
    batch_size: 28,
    max_audio_seconds: 15,
    max_decoder_seconds: 3,
    style_source: "quantized",
    prosody_source: "ground_truth",
    trainable_modules: ["alpha_flow"],
    enabled_losses: ["alpha_flow"],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: false,
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
