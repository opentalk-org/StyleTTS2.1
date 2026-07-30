export type ProsodySource = "ground_truth" | "predicted";
export type ReconstructionTarget =
  | "real_audio"
  | "teacher_reconstruction";
export type ValidationDurationSource = "ground_truth" | "predicted";
export type ValidationStageSpec = {
  f0_source: ProsodySource;
  norm_source: ProsodySource;
  duration_source: ValidationDurationSource;
  diffusion: boolean;
};

export const TRAINABLE_MODULES = [
  "bert",
  "bert_encoder",
  "decoder",
  "diffusion",
  "pitch_extractor",
  "predictor",
  "predictor_encoder",
  "style_encoder",
  "text_aligner",
  "text_encoder",
] as const;

export const TRAINING_LOSSES = [
  "adversarial",
  "diffusion",
  "duration",
  "duration_ce",
  "f0",
  "mel",
  "monotonic_alignment",
  "norm",
  "sequence_alignment",
  "slm_adversarial",
  "style",
  "wavlm",
] as const;

export type TrainableModule = (typeof TRAINABLE_MODULES)[number];
export type TrainingLoss = (typeof TRAINING_LOSSES)[number];
export type TrainingLossWeights = Record<TrainingLoss, number>;

export type TrainingStageSpec = {
  name: string;
  steps: number;
  prosody_source: ProsodySource;
  reconstruction_target: ReconstructionTarget;
  trainable_modules: TrainableModule[];
  enabled_losses: TrainingLoss[];
  loss_weights: TrainingLossWeights;
  train_discriminators: boolean;
  validation: ValidationStageSpec;
};

export const DEFAULT_LOSS_WEIGHTS: TrainingLossWeights = {
  adversarial: 1,
  diffusion: 1,
  duration: 1,
  duration_ce: 20,
  f0: 1,
  mel: 5,
  monotonic_alignment: 1,
  norm: 1,
  sequence_alignment: 1,
  slm_adversarial: 1,
  style: 1,
  wavlm: 1,
};
export const DEFAULT_VALIDATION: ValidationStageSpec = {
  f0_source: "predicted",
  norm_source: "predicted",
  duration_source: "ground_truth",
  diffusion: false,
};

const PREDICTORS: TrainableModule[] = [
  "bert_encoder",
  "bert",
  "predictor",
  "predictor_encoder",
];
const FINETUNE: TrainableModule[] = [
  ...PREDICTORS,
  "style_encoder",
  "decoder",
  "text_encoder",
  "text_aligner",
];
const PROSODY_LOSSES: TrainingLoss[] = [
  "mel",
  "f0",
  "norm",
  "duration",
  "duration_ce",
  "wavlm",
];
const FINETUNE_LOSSES: TrainingLoss[] = [
  ...PROSODY_LOSSES,
  "sequence_alignment",
  "monotonic_alignment",
  "adversarial",
];

export const STAGE_TEMPLATES: TrainingStageSpec[] = [
  {
    name: "First · acoustic bootstrap",
    steps: 100_000,
    prosody_source: "ground_truth",
    reconstruction_target: "real_audio",
    trainable_modules: ["text_encoder", "style_encoder", "decoder"],
    enabled_losses: ["mel"],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: false,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "First · TMA refinement",
    steps: 50_000,
    prosody_source: "ground_truth",
    reconstruction_target: "real_audio",
    trainable_modules: [
      "text_encoder",
      "style_encoder",
      "decoder",
      "text_aligner",
      "pitch_extractor",
    ],
    enabled_losses: [
      "mel",
      "sequence_alignment",
      "monotonic_alignment",
      "adversarial",
      "wavlm",
    ],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: true,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "Second · prosody bootstrap",
    steps: 100_000,
    prosody_source: "predicted",
    reconstruction_target: "teacher_reconstruction",
    trainable_modules: PREDICTORS,
    enabled_losses: PROSODY_LOSSES,
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: false,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "Second · style diffusion",
    steps: 50_000,
    prosody_source: "predicted",
    reconstruction_target: "teacher_reconstruction",
    trainable_modules: [...PREDICTORS, "diffusion"],
    enabled_losses: [
      ...PROSODY_LOSSES,
      "style",
      "diffusion",
      "adversarial",
    ],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: true,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "Second · joint",
    steps: 25_000,
    prosody_source: "predicted",
    reconstruction_target: "real_audio",
    trainable_modules: [
      ...PREDICTORS,
      "diffusion",
      "style_encoder",
      "decoder",
    ],
    enabled_losses: [
      ...PROSODY_LOSSES,
      "style",
      "diffusion",
      "adversarial",
      "slm_adversarial",
    ],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: true,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "Finetune · base",
    steps: 100_000,
    prosody_source: "predicted",
    reconstruction_target: "real_audio",
    trainable_modules: FINETUNE,
    enabled_losses: FINETUNE_LOSSES,
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: true,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "Finetune · diffusion",
    steps: 50_000,
    prosody_source: "predicted",
    reconstruction_target: "real_audio",
    trainable_modules: [...FINETUNE, "diffusion"],
    enabled_losses: [...FINETUNE_LOSSES, "style", "diffusion"],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: true,
    validation: { ...DEFAULT_VALIDATION },
  },
  {
    name: "Finetune · joint",
    steps: 25_000,
    prosody_source: "predicted",
    reconstruction_target: "real_audio",
    trainable_modules: [...FINETUNE, "diffusion"],
    enabled_losses: [
      ...FINETUNE_LOSSES,
      "style",
      "diffusion",
      "slm_adversarial",
    ],
    loss_weights: { ...DEFAULT_LOSS_WEIGHTS },
    train_discriminators: true,
    validation: { ...DEFAULT_VALIDATION },
  },
];

export const STAGE_PRESETS = [
  { label: "Original first", indexes: [0, 1] },
  { label: "Original second", indexes: [2, 3, 4] },
  { label: "Original finetune", indexes: [5, 6, 7] },
  { label: "Unified scratch", indexes: [0, 1, 2, 3, 4] },
] as const;

export function cloneStage(stage: TrainingStageSpec): TrainingStageSpec {
  return structuredClone(stage);
}
