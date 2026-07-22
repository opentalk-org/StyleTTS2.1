# Beetle Flow F0/N Ablation

## Goal

Determine whether Beetle's failed full synthesis is caused primarily by the
generated latent, by acoustic F0/N prediction from that latent, or by one-step
shortcut integration.

## Experiment

Use validation sample 1 from the active training run's step-8000 checkpoint.
Keep its conditioning, masks, waveform-generator seed, and initial flow noise
identical across variants. Generate three outputs:

1. One EMA flow step (`d=1`) with F0 and N extracted from the ground-truth mel.
2. 128 EMA flow steps (`d=1/128`) with F0 and N predicted from the sampled latent.
3. 128 EMA flow steps (`d=1/128`) with F0 and N extracted from the ground-truth mel.

The ground-truth-mel acoustic features come from Beetle's existing
`acoustic_targets` path. They replace only the F0 and N passed to the decoder;
the latent, decoder features, masks, and waveform generator otherwise remain
those of the corresponding flow sample.

## Execution and outputs

Run the diagnostic through Beetle's configured model, checkpoint, validation
loader, and synthesis components under the Nix development environment. Do not
alter the active trainer or normal validation behavior. Temporary diagnostic
code is removed after execution.

Save uncommitted WAV files and comparison plots under the active run output in
a diagnostic directory identifying checkpoint step 8000 and validation sample
1. Report the paths and compare the variants against the existing posterior and
one-step validation artifacts.

## Interpretation

- If GT F0/N rescues one-step audio, acoustic-feature collapse is a major
  downstream failure even when the one-step latent remains imperfect.
- If 128-step predicted F0/N works, shortcut distillation at large `d` is the
  primary failure.
- If only 128-step plus GT F0/N works, both large-step integration and acoustic
  prediction from sampled latents contribute.
- If none work, the base flow or generated latent distribution remains the
  dominant failure.
