# iSTFTNet2-MB Correction Design

## Goal

Make `src/runner/nodes/training/styletts3/testing/istftnet2_mb.py` implement the
iSTFTNet2-MB generator and synthesis geometry described by arXiv:2308.07117,
while retaining its standalone benchmark and repairing its optional NSF source.

## Architecture

The generator uses the HiFi-GAN V2 temporal front end with `C=128`, one `C4`
upsampling stage, and concatenated multi-receptive-field outputs. The resulting
192 channels are reshaped into 48 channels over a four-bin few-frequency axis.
Three independent MB ShuffleBlocks operate at that resolution. Three frequency
transposed convolutions then produce 8, 16, and 33 frequency bins while reducing
channels from 64 to 32 to 16 to the final eight magnitude/phase subband channels.

Each subband uses a 64-point Hann-window iSTFT with hop 16 and an explicit output
length. Four subbands are reconstructed through a deterministic 62-tap-prototype,
63-coefficient PQMF synthesis bank, producing exactly 256 waveform samples per
input mel frame.

The optional NSF source accumulates per-sample phase increments, handles voiced
and unvoiced frames, merges the fundamental and eight overtones, and injects the
source after temporal upsampling. Sampling rate is explicit rather than embedded
in the phase equation.

## Error Handling and Invariants

Constructor arguments must preserve the paper geometry: four bands, FFT size 64,
and channel/frequency dimensions divisible for the 1D-to-2D conversion. Tensor
reshapes should use declared dimensions rather than silent padding. iSTFT output
length and PQMF output length are explicit invariants.

## Validation

Use temporary tests that initially fail against the current implementation and
cover ShuffleBlock routing, frequency shapes, deterministic PQMF coefficients,
exact waveform length, and time-varying NSF output for constant voiced F0. Remove
the temporary tests before finishing. Run the standalone benchmark only after
these behavioral checks pass; inspect parameter count last and do not alter the
architecture to target the reported count.

## Scope

Only the standalone testing module is modified. No runner registration, training
pipeline, dependency, workflow, or persistent test changes are included.
