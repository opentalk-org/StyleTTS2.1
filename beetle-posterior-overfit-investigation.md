# Beetle Posterior Overfit Investigation

## Target and fixed test

The target is mean `train/posterior_reconstruction < 0.35` over steps
`950–999`, measured from a genuine step-zero run. Do not use a minimum,
hand-picked point, or preloaded model.

The fixed overfit dataset contains four validation recordings, each duplicated
16 times for 64 training rows. Validation uses the four unique recordings.
Batch size is 32 and seed is 4.

Recording IDs:

- `fa445cd1-e45b-4d04-9645-8e9c0d3403e4`
- `0a299823-3cb0-4bdb-98b2-797817a0238f`
- `5d261788-aa18-4a0a-ba59-9f925bbc0a0f`
- `a9437b86-d1cf-40fa-9773-7e3b205cbc81`

Do not change optimizer hyperparameters, configured loss weights, batch size,
or schedule to satisfy this test. Investigate architecture, objective routing,
and the training pipeline.

## Reference runs

| Run | Change | Comparison |
| --- | --- | --- |
| `df9c646951564cc0903a958e014a70a0` | Original baseline | Steps 950–999 mean `0.463713` |
| `1f8d258ba8254aef85d4d75deb9d72ff` | Linear latent-to-generator bypass | Steps 950–999 mean `0.444049` |

The retained linear bypass checkpoint is:

`src/runner/nodes/training/beetle/runs/overfit-validation-4x16-latent-bypass-b32-20260724/checkpoints/checkpoint_1e1901047eb94f468e5ba04cf65f642d`

## Completed ablations

| Run | Change | Result | Decision |
| --- | --- | --- | --- |
| `5bb74d3b93234721b1ef8f6e66815ac3` | Decoder LeakyReLU instead of Snake | Better through step 500, steps 950–999 `0.464263` | Reverted |
| `1f8d258ba8254aef85d4d75deb9d72ff` | Zero-initialized linear latent bypass | Steps 950–999 `0.444049`, 4.2% better | Retained |
| `a5c19e3324f44b21b8727063322a3b47` | Disable GAN/FM during reconstruction pretraining | Steps 451–500 `0.49508` vs bypass `0.49038` | Reverted |
| `c4e7071e53834110bb80bd3f02d82082` | Observe acoustic gradients without clipping | Steps 450–499 `0.51354` vs bypass `0.49053` | Reverted, stopped at 722 |
| `08dfbe0261ef498199446a94a185d0b7` | Stop F0/N auxiliary gradient at posterior latent | Steps 100–149 `0.67414` vs bypass `0.66405` | Reverted, stopped at 330 |
| `e28173312c8b4eb8a6f3a1b3cb679e88` | Use raw full-range iSTFT phase instead of `sin` | Steps 100–149 `0.69024` vs bypass `0.66405` | Reverted, stopped at 198 |
| `f6fc73ae1456433b88c37b620405aa15` | Feed log-Hz F0 to decoder projection | Steps 250–299 `0.55530` vs bypass `0.54691` | Reverted, stopped at 353 |
| `7240912cc36b412aa0101d5cdbe63292` | Nonlinear phase-distinct latent adapter | Steps 450–499 `0.52111` vs bypass `0.49053` | Reverted, stopped at 543 |
| `8df1f375251745c498cdeefa43e1dad1` | Preserve decoder identity shortcuts | Steps 450–499 `0.50717` vs bypass `0.49053` | Superseded by decoder replacement, stopped at 502 |

All stopped runs received SIGTERM and saved a cancellation checkpoint.

## Capacity localization

All capacity tests used the retained bypass checkpoint and the same four fixed
center crops.

- Optimizing the posterior latent through the frozen decoder and generator:
  `0.46347 → 0.41318` by 200 steps, then plateau.
- Optimizing the tensor entering the final decoder block:
  `0.42664 → 0.34465` by 1,000 steps.
- Optimizing generator frame features directly:
  `0.46347 → 0.33242` by 1,000 steps.
- Optimizing a linear transposed-convolution latent adapter:
  approximately `0.376` by 1,000 steps.
- Optimizing a contextual Snake + transposed-convolution latent adapter:
  `0.42664 → 0.34720` at 400, `0.33723` at 600, and `0.32927` at 1,000.

The nonlinear adapter proved sufficient frozen-model capacity but made joint
from-zero optimization worse. Capacity alone is not the current blocker.

## Disproved causes

- Posterior sampling noise is not the floor. Eight sampled latents averaged
  `0.436291`; posterior mean reconstruction was `0.436292`. Mean posterior
  noise scale was `0.131376`.
- F0 smoothing is not the floor. Kernel 7 changed loss by only `0.000697`;
  kernel 3 by `0.000044`.
- Waveform/crop alignment is not the floor. The best tested target shift was
  −30 samples and improved only `0.0004`, far below one 300-sample frame.
- Non-finite skipped steps are not the floor. Only six occurred before step
  1,000 in the inspected baseline.
- GAN and feature matching are not the primary early blocker. Removing them
  did not improve the step-500 trajectory.
- The three reconstruction resolutions do not conflict. At the predicted
  spectrum their gradient cosines are `0.889`, `0.782`, and `0.673`; at
  decoder features they are `0.904`, `0.794`, and `0.678`.
- The harmonic source is not hurting reconstruction. Checkpoint loss was
  `0.42660` normally, `0.43716` with source removed, and `1.18014` with only
  source.
- The phase `sin` has a restricted range but is not saturated in the retained
  checkpoint: phase-logit std is `0.0765`, range `[-1.234, 1.111]`, and 0% of
  values have `abs(cos(logit)) < 0.1`. Correcting the range slowed early
  convergence.
- Zero spectrum logits do map to zero waveform under the current
  polar/iSTFT/Hann overlap-add path. Do not replace the output representation
  based only on an assumed nonzero zero-state.
- Historical sub-0.35 runs before step 1,000 were not genuine step-zero runs.
  Their first logged steps were already approximately `0.28–0.33`.

## Gradient and activation dissection

At the retained step-1,000 bypass checkpoint, using reconstruction loss only:

| Location | Activation RMS | Gradient RMS |
| --- | ---: | ---: |
| Posterior latent | `18.5635` | `1.62e-4` |
| Final decoder features | `5.1469` | `3.61e-5` |
| Generator input projection | `22.9642` | `6.75e-5` |
| Temporal upsample | `4.1334` | `1.93e-4` |
| Frequency entry | `3.4909` | `1.38e-4` |
| Final spectrum | `0.1469` | `3.55e-2` |

Weighted reconstruction parameter-gradient norms were:

- audio encoder: `9.52`
- decoder: `11.70`
- generator: `70.28`

The fresh model showed generator norm `71.92`, so the large generator gradient
is inherent to this path rather than a late GAN effect.

The posterior latent effective channel rank collapses from `30.97` fresh to
`1.42` at step 1,000, with the largest direction carrying `83.3%` of energy.
The first decoder block is also low rank at initialization (`1.56`, largest
direction `80.0%`).

Raw decoder input scales at initialization were:

- latent residual RMS: `0.686`
- projected N RMS: `2.181`
- projected F0 RMS: `66.339`

At step 1,000 they were `5.468`, `0.880`, and `10.311`. The model compensates
for raw-Hz F0 dominance by shrinking its projection and inflating the latent.
Log-Hz conditioning fixed this scale mismatch but did not improve the measured
training trajectory, so do not repeat that isolated ablation.

F0+N auxiliary gradients into the audio encoder had norm `16.84`, versus
weighted reconstruction norm `9.52`, with cosine `−0.009`. Detaching the
auxiliary head did not improve early training, so this conflict is real but
not sufficient to explain the floor.

Independent module clipping continuously rescales ordinary gradients, but
removing acoustic clipping made reconstruction worse. Do not repeat the
all-unclipped ablation. AMP gradients are unscaled before these measurements.

Decoder reconstruction parameter-gradient norms split by branch were:

- encoder block: `0.407`
- decode blocks 0–3: `0.244`, `0.309`, `0.530`, `1.593`
- repeated latent residual: `6.038`
- linear generator bypass: `9.553`
- F0 projection: `2.443`
- N projection: `0.0027`

The residual block returned `(shortcut + residual) / sqrt(2)`, attenuating the
identity route by `1/sqrt(2)` per block and by `1/4` across four blocks. A
zero-residual identity test failed with exactly the expected `0.292893`
relative error. The active test change preserves the shortcut exactly and
scales only the learned residual: `shortcut + residual / sqrt(2)`.

## Per-sample checkpoint losses

The four fixed crops at the retained checkpoint gave:

| Recording | Loss | Voiced ratio |
| --- | ---: | ---: |
| `fa445cd1…` | `0.39372` | `0.469` |
| `0a299823…` | `0.41644` | `0.969` |
| `5d261788…` | `0.47732` | `1.000` |
| `a9437b86…` | `0.43993` | `1.000` |

The floor is shared by all samples rather than caused by one outlier.

## Integrated replacement

Isolated ablations can hide compounding architecture faults. Proven compatible
corrections should be tested together rather than automatically reverted when
each is insufficient alone.

The active integrated replacement removes the old 30M-parameter low-rate
decoder and its repeated conditioning concatenation. The replacement has:

- a learned `ConvTranspose1d` from the 192-channel half-rate latent directly
  to 512 frame-rate generator features;
- frame-rate `log1p(F0)` and N conditioning projected into the same space;
- four identity-preserving gated dilated residual blocks at dilations
  `1, 3, 9, 27`;
- 7.75M parameters;
- F0/N auxiliary gradients stopped at the posterior latent;
- unrestricted phase angles at `torch.polar`.

Clipping, optimizer settings, GAN schedule, configured loss weights, and data
settings remain unchanged because changing them either regressed or would
confound the architectural test.

## Active state and next diagnostic

No training process is running. The integrated frame-rate decoder has a passing
shape, conditioning, and F0-preservation test. It has not yet been evaluated in
a from-zero run.

## Integrated decoder run

Run `3b1a8080cf76489894f65782601c9188` used the integrated frame-rate
decoder with the original 1,000-step optimizer warmups:

- steps 100–149: `0.63266` vs prior best `0.66405`;
- steps 250–299: `0.50480` vs prior best `0.54691`;
- steps 450–499: `0.45016` vs prior best `0.49053`;
- steps 750–799: `0.44882` vs prior best `0.46714`.

It was stopped at step 874 after plateauing. Checkpoint:

`src/runner/nodes/training/beetle/runs/overfit-validation-4x16-frame-decoder-b32-20260724/checkpoints/checkpoint_c8c5292e92704725871854bcb0bea7a1`

The decoder replacement is retained. The next user-specified test changes both
generator and discriminator optimizer LR warmups from 1,000 to 100 steps.
The first attempt was stopped at step 18 because its process loaded the config
before the subsequent edit. The complete user-specified schedule also changes
discriminator, generator-adversarial, feature-matching, and style-adversarial
loss warmups from 1,000 to 100 steps. Test the complete schedule from zero.

## Four-source result was not the intended test

Run `290bf2d8338b4fa996162c1c68629216` combined the integrated frame-rate
decoder with 100-step optimizer and GAN-related loss warmups.

`train/posterior_reconstruction` over steps 950–999:

- count: `50`
- mean: `0.338333`
- standard deviation: `0.029619`
- minimum: `0.296351`
- maximum: `0.400232`

This passed the numerical `< 0.35` last-50-step target on four unique sources,
but it did not test the intended scope. Step-1,000 validation reconstruction
was `0.430833`, and the reconstructed audio was perceptually unusable. It is
therefore not evidence that the posterior architecture is fixed.

The run was stopped cleanly at step 1,070. Cancellation checkpoint:

`src/runner/nodes/training/beetle/runs/overfit-validation-4x16-frame-decoder-all-warmup100-b32-20260724/checkpoints/checkpoint_d3b38036c1174177bd5f84c06e4629f7`

No training process is running. No commit was created.

The subsequent all-16-source experiments and acoustic-conditioning diagnosis
are recorded in `beetle-posterior-overfit-all16-investigation.md`.
