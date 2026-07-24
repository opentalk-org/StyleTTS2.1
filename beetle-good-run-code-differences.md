# Beetle Good-Run Code Difference Ledger

Reference MLflow run: `46f0cfd4b193457ba679ff4e635d3da1`

Reference Beetle source: `7ceedba`. The run started before that commit was
created, but its logged configuration and posterior diagnostics prove that the
dirty working tree already contained the Beetle changes later recorded there.
The rebased Beetle-equivalent commit is `58b8645`.

Current comparison command:

```bash
./nix/run-venv.sh git diff 7ceedba -- src/runner/nodes/training/beetle
```

## Experiments

### Restored waveform path

The working tree restores these post-reference changes:

- `65b5d73`: Snake generator residual activations are replaced by the reference
  LeakyReLU residual activations.
- `9be9c62`: the spectral harmonic-source adapter is removed.
- `9be9c62`: iSTFT log-magnitude clipping is removed.
- `8509725`: reconstruction loss again aggregates numerator and denominator
  across equal-length batch groups, matching the reference objective.

Result at step 500:

- feature matching: `9.35`, reference `7.21`
- discriminator gradient: `162.4`, reference `30.6`
- decoder gradient: `7.42`, reference `55.8`

The waveform restoration helped the discriminator relative to the preceding
run, but did not reproduce the reference trajectory.

### Eager-generator experiment

This experiment removed only the generator from `compile_acoustic`.
Feature-linear, decoder-block, and discriminator compilation remained unchanged.

Reason: before the first optimizer update, decoder gradients are approximately
`1.5` in the restored-waveform run and `18.8` in the reference, while generator
parameter gradients are much closer. The reference left the generator eager;
`8509725` compiled it as a whole module.

Result: disproven. With the generator eager, decoder gradients are `1.72` at
step 1 and `1.60` at step 10, still far below the reference `18.73` and `18.86`.
The run was stopped after collecting this evidence, and generator compilation
was restored.

### Accelerator mixed-precision experiment

`8509725` changed Accelerator from `mixed_precision="no"` to `"bf16"` for the
active configuration, while retaining Beetle's explicit outer
`torch.autocast(...)` context.

Accelerate source inspection and an isolated runtime check show that this is
not redundant bookkeeping. `prepare_model()` replaces every separately
prepared module's `forward` with another autocast wrapper and converts BF16
outputs to FP32. Current training therefore inserts FP32 boundaries after the
audio encoder, feature-linear layer, decoder, generator, and discriminators.
The reference run kept intermediate outputs under the one outer Beetle
autocast.

The ablation restores `mixed_precision="no"` in Accelerator without changing
Beetle's configured BF16 precision or its explicit autocast.

Early result initially appeared supportive. The decoder gradient rose from approximately
`1.5`-`1.7` in the broken runs to `11.25` at step 1 and `11.57` at step 10.
The reference values are `18.73` and `18.86`. The generator gradient is
`74.00` at step 1 versus the reference `66.10`, and the audio-encoder gradient
is `6.87` versus `7.66`. This restores most of the missing initial acoustic
backward signal.

However, the later eager-discriminator run retained the precision change and
produced decoder gradients of only `2.20` and `2.17` at steps 1 and 10.
Because fresh model initialization is unseeded, the early-gradient comparison
is confounded and does not independently prove the precision change's training
effect. What is proven is the FP32-boundary execution mismatch; a seeded paired
A/B is required to quantify its effect.

The later result is only a partial fix. At step 200, feature matching and
discriminator loss nearly match the reference (`5.83` versus `5.53`, and
`3.02` versus `3.03`). After that, the current discriminator becomes unstable:
at step 500 feature matching is `11.19` versus `7.21`, discriminator gradient
is `228.83` versus `30.58`, and reconstruction is `0.540` versus `0.497`.
The run `d22625a127f74fa389a2e98cec85b6c0` was stopped after collecting the
step-500 comparison.

### Eager-discriminator experiment

The reference leaves all discriminators eager. Current code compiles the
combined discriminator `forward`, then calls it alternately with parameters
trainable for the discriminator step and temporarily frozen for the generator
step. Because the precision ablation tracks the reference until adversarial
warmup becomes substantial and then diverges sharply, this experiment removes
only discriminator compilation. Channels-last layout, critic topology, losses,
segment length, and all other compilation remain unchanged.

Result: supported. Run `7d06e2b690994997b81b4a509638d21f` reached
step 500 with discriminator gradient `19.97`, compared with `228.83` when
compiled and `30.58` in the reference. Feature matching improved from `11.19`
to `9.63` (reference `7.21`). Generator gradient was `255.05`, close to the
reference `238.25`.

The remaining mismatch has the critic winning too strongly: discriminator
loss is `2.27` versus the reference `2.79`, while reconstruction is `0.552`
versus `0.497`. Current generator peak LR remains one third of the reference
while discriminator LR matches, which is a direct remaining optimizer-balance
difference.

### Reference generator-LR experiment

With discriminator execution restored, the next ablation changes only
generator peak LR from `2e-4` to the reference `6e-4`. Discriminator LR,
optimizer betas, decay, all loss schedules, model code, data, and segment
geometry remain unchanged.

This experiment is pending a fresh step-500 run.
At step 100, the trajectory is close to the reference: reconstruction `0.726`
versus `0.709`, feature matching `2.322` versus `2.314`, discriminator loss
`4.745` versus `4.532`, discriminator gradient `0.479` versus `0.418`, decoder
gradient `13.46` versus `17.87`, and generator gradient `49.28` versus `60.39`.

At step 500, reconstruction is `0.49788` versus the reference `0.49739`.
Across steps 450-500, mean reconstruction is `0.48491` versus `0.48480`,
discriminator gradient is `22.21` versus `22.15`, decoder gradient is `28.00`
versus `31.48`, and audio-encoder gradient is `63.46` versus `68.43`. Mean
feature matching remains higher at `8.46` versus `7.26`.

The same configuration was rerun from zero through optimizer-complete step
1504 as MLflow run `f36ce42257b643928948baff3cb6dd58`, with validation at
step 1400. It completed validation and reported no skipped or nonfinite steps.

Across steps 1450-1500, current versus reference means are:

- reconstruction: `0.39212` versus `0.39300`
- feature matching: `5.71090` versus `5.66662`
- discriminator loss: `3.11080` versus `3.13465`
- discriminator gradient: `29.83` versus `41.10`
- decoder gradient: `17.52` versus `16.38`
- generator gradient: `107.33` versus `156.15`

The step-1400 validation aggregate is reconstruction `0.37891`, feature
matching `4.96305`, discriminator loss `3.48458`, duration flow `1.35132`, and
latent flow `11.14816`. This confirms that the restored configuration remains
stable through the interval where the compiled-discriminator run had already
diverged.

### FP16 and generator-path ablations

Run `a11114c3c4e04bff84edf9df8d489cdc` restored Snake activations and enabled
FP16 both in Beetle's explicit autocast/scalers and in Accelerator, while
leaving the spectral source adapter absent. It remained stable through step
989, when it was stopped for the next ablation. At step 500, reconstruction
was `0.48361`, feature matching `8.11456`, discriminator loss `2.58551`,
decoder gradient `46.71`, and generator gradient `244.97`. The generator AMP
scale had recovered from `16` to `8`, with no skipped loss steps.

Run `a14b18d7323a44cc8bfd29a39e99494d` additionally restored
`SpectralSourceAdapter`. At step 500 its GAN metrics were closer to the
reference: reconstruction `0.50669`, feature matching `7.45765`,
discriminator loss `2.90962`, and discriminator gradient `27.35`. Its decoder
gradient was lower at `18.62`, and its generator AMP scale had fallen to `2`.

The spectral run advanced stably through optimizer step 1232, then every
subsequent generator batch produced nonfinite metrics. It skipped 51
consecutive batches without advancing and was stopped before validation. The
finite discriminator losses during the skipped batches show that failure was
generator-side. This does not isolate the adapter as the sole cause: the
otherwise identical no-adapter FP16 run was replaced at step 989 and therefore
did not reach the failure point. It does show that the final FP16 + Snake +
spectral configuration is not viable through 1500 steps as currently
implemented.

## Remaining acoustic-path differences

### Compilation

Reference:

- compiles the audio encoder, feature-linear layer, and decoder blocks with
  `nn.Module.compile()`
- leaves generator and discriminators eager

Current eager-discriminator experiment:

- leaves the audio encoder eager
- compiles feature-linear and decoder blocks by assigning
  `torch.compile(..., dynamic=False)` to each concrete `forward`
- compiles the generator as a whole model
- leaves discriminators eager

### Discriminators

`8509725` converts spectrogram- and period-discriminator convolution weights,
inputs, and intermediate activations to channels-last memory format. Layer
topology and LSGAN/feature-matching formulas remain unchanged.

### Synthesis and window selection

Reference Stage 1:

- the data loader plans one 9,600-sample target window and its encoder context
- discriminator and generator steps consume that preplanned target window
- decoder receives ground-truth F0 and N

Current:

- trainer independently selects discriminator and generator segments from each
  collated utterance
- `synthesize_training_posterior` extracts encoder context around each selected
  segment and slices the posterior afterward
- decoder receives ground-truth F0 and N while acoustic prediction ratio is zero
- adversarial segments are 19,200 samples by explicit experiment constraint

### Data pipeline

`8509725` replaces `stage1_loader.py`, `stage1_records.py`, and
`stage1_sampling.py` with the shared grouped batch pipeline. Current constraints
intentionally retain this implementation.

Configuration differences retained intentionally:

- maximum audio duration: 8 seconds instead of 45
- no 1-second minimum
- time stretch and pitch shift disabled
- current grouped sampling and pseudo-speaker metadata
- 19,200-sample adversarial segment

### Optimizer and schedules

- reference and current experiment generator peak LR: `6e-4`
- reference generator decay: 500,000 steps; current: 1,000,000
- discriminator optimizer settings match
- both implementations clip named acoustic modules independently
- current generator optimizer also owns conditional modules, but each module
  group is clipped separately
- conditional audio-encoder reads use `torch.no_grad()`, so conditional losses
  do not backpropagate into the shared audio encoder

### Acoustic losses

- reconstruction aggregation is restored to the reference implementation
- KL currently sums 192 latent channels before masked averaging; the reference
  averaged channels
- KL weight is zero in the active ablations, so this reduction cannot affect
  their acoustic gradients
- F0, N, adversarial, and feature-matching formulas match the reference

### Unified conditional training

`8509725`, `2aa4731`, and `68109ee` replace isolated Stage 1 with the unified
trainer. Conditional losses execute every generator step. They share an
optimizer but do not share acoustic parameter gradients except through
explicitly shared modules; audio-encoder extraction is detached as noted above.

### Reproducibility

Neither the reference nor current training entry point seeds the process-wide
PyTorch or Python random generators before model construction. The configured
runtime seed is used for explicit data, segment, posterior-noise, and source
generators, but not for initial model weights, dropout, or the decoder's
`random.choice` smoothing kernels. Exact step-for-step reproduction across
fresh processes is therefore impossible; early values must be compared as
trajectories rather than expected to match bit-for-bit.

Unified conditional training also consumes process-wide dropout randomness
between acoustic steps. It cannot add a conditional gradient to the detached
audio encoder, but it changes the later acoustic dropout sequence relative to
isolated Stage 1.

### Runtime and failure handling

- `8509725` enabled Accelerate-managed mixed precision on top of Beetle's
  explicit autocast; the current ablation restores the reference runtime
  precision ownership
- current training skips nonfinite steps and reports `skipped_steps`
- current checkpoint/reporting lifecycle differs from staged training
- these paths do not alter a finite acoustic forward/backward step

## Commit sequence after the reference

- `65b5d73`: Snake generator activations
- `9be9c62`: spectral source path and conditioning-pyramid changes
- `8509725`: unified trainer, shared data pipeline, 19,200-sample windows,
  reconstruction reduction rewrite, generator/discriminator compilation, and
  channels-last discriminators
- `2aa4731`: posterior-only acoustic waveform training
- `68109ee`: aligned-window and conditional-input restructuring
- `bd4299d`: configuration changes
- `c3d093b`: KL channel sum, nonfinite-step handling, metric changes, and removal
  of the complexity gate
