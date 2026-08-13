# StyleTTS training comparison with StyleTTS-ZS

Paper: [StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis with Distilled Time-Varying Style Diffusion](https://arxiv.org/pdf/2409.10058)

## Verdict

The current trainer is a substantial StyleTTS2/StyleTTS-ZS hybrid, but it is not a faithful implementation of the paper's training method. The prosody autoencoder and residual vector quantizer are reasonably close; the paper's central diffusion-distillation method is not implemented.

| Area | Current implementation | Paper fidelity |
|---|---|---|
| Prompt/text encoder | Conformer-based, prompt-aligned representations | Partial |
| Fixed 50-token prosody latent | Implemented | Strong |
| RVQ: 9 x 1024, dimension 8 | Implemented | Strong |
| Prosody adversarial training | Implemented | Partial/strong |
| Speaker feature matching | Implemented | Strong |
| Optimizer hyperparameters | AdamW `(0, .99)`, weight decay `1e-4`, learning rate `1e-4` | Strong |
| Multimodal waveform discriminator | WavLM discriminator lacks decoder-input conditioning | Weak |
| Teacher style diffusion | Replaced by an AlphaFlow/MeanFlow-style objective | Missing |
| DDIM-100 and CFG teacher | Not present | Missing |
| One-step student distillation | Not present | Missing |
| 10k teacher-pair dataset and perceptual distillation | Not present | Missing |
| Paper-scale training | Default is 43k fine-tuning steps | Major mismatch |

## Critical differences

### 1. Diffusion and distillation are fundamentally different

The paper trains angular-schedule velocity prediction, creates targets using a 100-step DDIM teacher with classifier-free guidance, and distills that teacher into a one-step student using an L1 perceptual loss after the prosody decoder.

Locally, the final stage trains `AlphaFlow` using a mixture of flow-matching and MeanFlow-like targets:

- [`default_stages.py`](src/runner/nodes/training/styletts/finetune/default_stages.py), lines 145-154
- [`alpha_flow.py`](src/runner/nodes/training/styletts/finetune/training/modules/latent/alpha_flow.py), lines 43-144

There is no:

- 100-step DDIM teacher
- CFG-conditioned teacher generation
- stored or generated set of 10,000 teacher pairs
- separate distilled student
- guidance scale sampled from `[1, 15]`
- prosody-decoder perceptual distillation objective

Consequently, the implementation cannot reproduce the paper's one-step distilled model or its reported speed claims.

### 2. The flow model receives the continuous RVQ latent

The trainer computes the RVQ output but passes `continuous_latent.detach()` into AlphaFlow:

- [`trainer.py`](src/runner/nodes/training/styletts/finetune/training/runtime/trainer.py), lines 218-247

This weakens the paper's central rationale for RVQ: simplifying and regularizing the style distribution before diffusion and distillation. The exact intended latent interface should be reconciled with Sections III-B and III-C of the paper before changing it, but the current path does not clearly model the quantized representation.

### 3. The multimodal waveform discriminator is incomplete

The paper conditions its WavLM-based waveform discriminator on prompt-aligned text, global style, pitch, energy, and duration. The local WavLM discriminator receives speech features without those decoder conditions:

- [`discriminators.py`](src/runner/nodes/training/styletts/finetune/training/modules/discriminators.py), lines 227-244
- [`trainer.py`](src/runner/nodes/training/styletts/finetune/training/runtime/trainer.py), lines 593-616

The MPD and multi-resolution spectral discriminators are present, so this is a missing conditional critic rather than a total absence of waveform GAN training.

### 4. Global-style dropout differs from the paper

The paper drops global style 20% of the time so the prompt-aligned text representation must retain prompt information. Locally, training randomly suppresses either voice conditioning or prompt-aligned text:

- [`trainer.py`](src/runner/nodes/training/styletts/finetune/training/runtime/trainer.py), lines 319-330

That changes the regularization objective. The local global style is also 128-dimensional, while the paper specifies a pooled 512-dimensional representation:

- [`base.yaml`](src/runner/nodes/training/styletts/finetune/data/base.yaml), line 24
- [`voice.py`](src/runner/nodes/training/styletts/finetune/training/modules/latent/voice.py), lines 40-61

### 5. Training scale and data policy differ substantially

The local default schedule is 43,000 steps:

1. 10k StyleTTS2 mel pretraining
2. 10k acoustic GAN
3. 5k continuous prosody
4. 8k RVQ prosody
5. 10k AlphaFlow

See [`default_stages.py`](src/runner/nodes/training/styletts/finetune/default_stages.py), lines 98-155.

The paper trains:

- LibriTTS for 30 epochs, or LibriLight for 1,000,000 steps
- full utterances up to 30 seconds for prosody and diffusion
- waveform segments up to 3 seconds
- a distilled student for another 10 epochs over 10,000 generated pairs

Local stage values of 65-150 seconds are dynamic batch audio budgets, while decoder crops are 3-9 seconds. These controls do not enforce the paper's exclusion of utterances shorter than 1 second or longer than 30 seconds.

## Objective mismatches

- The paper uses L1 prosody reconstruction; local training uses length-scaled Smooth L1.
- The paper gives pitch weight `0.1`; local configuration uses `1`, although the local F0 calculation divides by 10, producing a roughly similar effective scale with different gradients.
- The local `mel` loss is a multi-resolution normalized log-mel/STFT waveform loss rather than the paper's direct mel-spectrogram L1.
- Alignment defaults differ: local sequence/monotonic weights are `1/10`; the paper reports `0.2/5`.
- OneCycle schedulers are constructed locally but apparently never stepped and are not checkpointed: [`optimizers.py`](src/runner/nodes/training/styletts/finetune/training/optimizers.py), lines 68-86.

## Strong and partial matches

The strongest paper-aligned portion is the prosody representation:

- positional embeddings plus pitch and energy
- fixed `50 x 512` latent
- Conformer-based encoder
- nine-stage residual vector quantization
- 1,024 entries per codebook
- eight-dimensional code projections
- six-layer prosody prediction stack
- multimodal prosody discriminator and feature matching
- WeSpeaker intermediate-feature matching

Relevant implementation:

- [`prosody.py`](src/runner/nodes/training/styletts/finetune/training/modules/latent/prosody.py)
- [`rvq.py`](src/runner/nodes/training/styletts/finetune/training/modules/latent/rvq.py)
- [`features.py`](src/runner/nodes/training/styletts/finetune/training/features.py)

Other partial matches include 24 kHz audio, 80-bin mel inputs, FFT 2048, window 1200, hop 300, prompt/text Conformers, prosody masking, waveform MPD/MR-STFT critics, and the paper's AdamW hyperparameters.

## Recommended priority

1. Implement the paper's teacher diffusion and explicit one-step distillation pipeline.
2. Establish whether diffusion targets the quantized RVQ representation or its decoded continuous form, then make that interface explicit.
3. Add the decoder-conditioned multimodal WavLM discriminator.
4. Correct global-style-only dropout and decide whether checkpoint compatibility justifies retaining the 128-dimensional style projection.
5. Align dataset duration filters, training horizon, L1 objectives, and published loss weights.
6. Add evaluation for WER, speaker similarity, RVQ utilization, teacher-student prosody error, CFG behavior, and held-out speakers.

## Paper details relevant to reproduction

- Prosody encoder: one kernel-31 Conformer followed by five kernel-15 Conformers, eight heads, 512 hidden dimensions, then cross-attention into 50 fixed positions.
- RVQ: nine codebooks, 1,024 entries per codebook, and an eight-dimensional quantization projection.
- Prosody decoder: six Conformers with separate duration and pitch/energy heads.
- Teacher diffusion: angular schedule, L1 velocity objective, condition dropout `0.1`, CFG guidance `5`, and 100 deterministic DDIM steps.
- Distillation: 10,000 teacher-generated pairs, guidance sampled uniformly from `[1, 15]`, perceptual L1 after the prosody decoder, and 10 student epochs.
- Acoustic training: L1 mel reconstruction, ASR sequence and monotonic alignment losses, waveform adversarial and feature-matching losses, and speaker-verifier feature matching.
- Data: LibriTTS or LibriLight, 24 kHz, phonemized text, utterances between 1 and 30 seconds, batch size 32.

This report is a read-only comparison of the repository implementation against the cited paper. No runtime validation or training run was performed.
