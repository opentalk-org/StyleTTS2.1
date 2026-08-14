# StyleTTS training comparison with StyleTTS-ZS

Paper: [StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis with Distilled Time-Varying Style Diffusion](https://arxiv.org/pdf/2409.10058)

## Verdict

The current trainer is a substantial StyleTTS2/StyleTTS-ZS hybrid, but it is not a faithful implementation of the paper's training method. The prosody autoencoder and residual vector quantizer are reasonably close; the paper's central diffusion-distillation method is not implemented.

| Prompt/text encoder | Conformer-based, prompt-aligned representations | Partial |

| Prosody adversarial training | Implemented | Partial/strong |

generated:
A
A
A
B


## Critical differences

- Alignment defaults differ: local sequence/monotonic weights are `1/10`; the paper reports `0.2/5`.


## Recommended priority



5. Add evaluation for WER, speaker similarity, RVQ utilization, teacher-student prosody error, CFG behavior, and held-out speakers.
