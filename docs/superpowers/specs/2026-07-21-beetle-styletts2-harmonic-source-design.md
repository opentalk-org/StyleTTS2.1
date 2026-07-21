# Beetle StyleTTS2 Harmonic Source

## Goal

Make Beetle construct its harmonic excitation with the same phase and gradient
contract as the working StyleTTS3 iSTFTNet2-MB path while retaining Beetle's
seeded randomness, masks, and generator geometry.

## Design

Beetle will construct the nine harmonic components in float32 regardless of
autocast precision. It will follow StyleTTS2's frame-rate phase algorithm:
expand frame F0 to sample rate, form harmonic phase increments, reduce those
increments back to frame rate, accumulate phase at frame rate, scale the phase
by the output hop, and linearly interpolate it to sample rate. The fundamental
starts at zero phase and the overtones receive seeded random initial phases.

Sine and noise construction will not retain a gradient to the predicted F0.
The resulting excitation remains an input to Beetle's learned harmonic merge,
source projection, source residual, and spectrogram generator, all of which
remain trainable. F0 also remains differentiable through Beetle's separate
decoder-conditioning path and retains its supervised F0 objective.

The implementation stays inside the Beetle node family. It will not import the
StyleTTS2 or StyleTTS3 implementation modules.

## Scope

Only the harmonic-source construction changes. Reconstruction losses,
optimizer settings, generator topology, PQMF/iSTFT synthesis, masking, and
training configuration remain unchanged.

## Validation

Temporary tests will establish that:

- bfloat16 F0 produces float32 harmonic construction with the configured output
  length;
- the excitation has no gradient dependency on F0;
- the learned harmonic merge still receives gradients;
- equal seeds produce equal excitation and different seeds alter stochastic
  components;
- constant F0 produces the expected fundamental frequency without the prior
  bfloat16 cumulative-phase corruption;
- the complete Beetle generator retains its waveform geometry.

Repository policy requires removing temporary tests before completion. Focused
checks will run through `nix develop --command`.
