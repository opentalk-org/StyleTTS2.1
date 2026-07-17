# iSTFTNet2-MB Paper-256 Training Profile Design

## Goal

Add a paper-reference iSTFTNet2-MB training profile under `styletts3` so the
existing native-hop-300 experiment can be compared with the model geometry and
signal preprocessing described by Kaneko et al. The reference profile uses the
project's StyleTTS GAN backend by explicit request; that discriminator and its
relative adversarial term are the intentional deviation from the paper.

## Command Boundary

The existing training entry point gains a typed profile option:

```bash
nix develop --command python -m runner.nodes.training.styletts3.testing.train_istftnet2_mb \
  --profile paper_256 \
  --discriminator styletts \
  --dataset-id <uuid> \
  --output-dir <path>
```

The current native-hop-300 profile remains the default and retains its current
behavior. The paper profile defaults to the StyleTTS discriminator and rejects
another discriminator selection so the requested experiment is reproducible.

## Generator Architecture

The paper profile implements iSTFTNet2-MB as `C4-I16-B4`:

- 80-channel mel input and HiFi-GAN V2 initial width `C=128`;
- one factor-four temporal transposed convolution;
- three 1D residual branches with kernels 3, 7, and 11, concatenated before
  conversion into a four-bin frequency axis;
- three paper-modified 2D ShuffleBlocks whose active branch maps `C/2 -> C -> C/2`;
- frequency expansion `4 -> 8 -> 16 -> 33` while channels reduce
  `64 -> 32 -> 16 -> 8`;
- eight output channels representing magnitude and phase for four subbands;
- a 64-point Hann-window iSTFT with hop 16 per subband;
- four-band PQMF synthesis, yielding exactly 256 waveform samples per mel frame.

The hop-300 implementation is not rewritten to use the new geometry. Shared
blocks may be reused where their behavior is identical, while profile-specific
iSTFT and frequency-expansion details remain explicit.

## Signal and Data Profile

The paper profile caches resampled copies at 22,050 Hz without changing backend
audio objects or cached files belonging to the native profile. Training uses
8,192-sample segments and batches of 16. Conditioning uses 80-bin log-mels with
FFT 1024, Hann window 1024, hop 256, `fmin=80`, and `fmax=7600`.

The reconstruction objective uses the linked ParallelWaveGAN HiFi-GAN mel
settings: FFT/window 1024, hop 256, 80 bins, `fmin=0`, and `fmax=11025`. Signal
geometry is carried through a frozen typed profile rather than mutable module
globals. Audio-cache keys include the target sample rate so the two profiles do
not alias each other's files.

## Optimization

The paper profile defaults to 2.5 million optimizer steps, batch size 16, Adam
learning rate `2e-4`, and betas `(0.5, 0.9)`. The generator objective is:

`45 * mel + adversarial + 2 * feature_matching`

The discriminator is the existing StyleTTS combination of multi-period and
multi-resolution spectrogram discriminators. Its existing least-squares and
relative adversarial behavior is preserved. This is recorded in MLflow as an
explicit paper deviation.

`--max-steps` provides a bounded smoke path without changing the normal paper
target. The native profile keeps its epoch-based behavior. The trainer stops at
the first satisfied bound and writes the existing final generator artifact and
validation outputs.

## Reporting and Failure Behavior

MLflow records the selected profile, sample rate, conditioning and synthesis
geometry, optimizer settings, loss weights, target steps, and the StyleTTS GAN
deviation. Invalid profile/discriminator combinations and tensor geometry fail
with explicit errors. No fallback silently changes sample rate, hop, generator,
or discriminator.

## Verification and Runtime Safety

Temporary tests cover CLI selection, profile-specific cache paths and mel
geometry, paper generator intermediate/output shapes, and exact 256x output.
They also ensure the native profile remains the default. Tests are written first,
observed failing, run through `nix develop --command`, and removed before
completion as required by the repository.

A bounded CPU forward/backward integration check exercises the paper generator
and StyleTTS GAN without launching a CUDA training job. The active hop-300 process
is never signaled, restarted, or replaced. Existing uncommitted edits are
preserved, and implementation verification does not start another accelerator
workload.
