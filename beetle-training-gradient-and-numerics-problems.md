# Beetle Training Gradient and Numerics Problems

## Summary

The current training path contains two separate concerns:

1. Ordinary finite gradients are clipped far too frequently to serve as
   anomaly protection.
2. FP16 numerical failures are skipped with insufficient diagnostics.

The same clipping policy was present in the older good run, so it does not
explain the quality regression by itself. It is nevertheless a poor baseline
for future optimization and may prevent better convergence.

## Current gradient path

For each optimizer step:

1. The discriminator loss is backpropagated with its FP16 `GradScaler`.
2. The acoustic and conditional generator losses are backpropagated into the
   shared generator optimizer with a separate FP16 `GradScaler`.
3. Each scaler unscales its optimizer's gradients.
4. The complete optimizer gradient norm is measured for finiteness and
   telemetry only.
5. Every named module group except `feature_linear` is independently clipped
   to norm `10`.
6. AdamW performs its normal moment-normalized update and weight decay.

There is no optimizer-global generator clipping after the per-module groups
were introduced. A large latent-flow gradient does not directly reduce the
waveform generator gradient. Any note claiming that latent flow consumes one
shared global clipping budget is incorrect for the current code.

## Continuous clipping instead of anomaly clipping

In the inspected current run through approximately step 3,700:

| Gradient group | Fraction above norm 10 | Median raw norm |
| --- | ---: | ---: |
| generator | 100% | 141 |
| latent flow | 100% | 3,432 |
| conditioning | 98% | 66 |
| audio encoder | 94% | 20 |
| context encoders | 93% | 29 |
| decoder | 90% | 18 |
| discriminators | 88% | 25 |
| phoneme encoders | 82% | 20 |
| duration predictor | 64% | 13 |

`feature_linear` is observe-only. The aligner and most style/voice groups are
usually below the cap.

This means the norm-10 policy is routinely replacing the natural gradient
amplitude with norm `10`. It is not merely catching rare explosions.

The older good BF16 run had similar first-3,750-step clipping rates:

| Gradient group | Good run | Current run |
| --- | ---: | ---: |
| audio encoder | 93.0% | 93.9% |
| decoder | 83.0% | 89.8% |
| generator | 100% | 100% |
| discriminators | 89.7% | 88.2% |

Therefore:

- continuous clipping is not sufficient to explain why the current run sounds
  worse;
- the good run does not prove the clipping policy is optimal;
- replacing it should be evaluated as a forward optimization improvement.

## Is gradient scaling equal to loss scaling?

For an ordinary differentiable loss `L` and a constant scalar `c`:

```text
gradient(c * L) = c * gradient(L)
```

Therefore, before clipping and before the optimizer, multiplying the complete
loss by `c` is mathematically equivalent to multiplying every gradient
produced by that loss by `c`.

### Scaling one component of a combined loss

For:

```text
total = a * reconstruction + b * adversarial
```

changing `a` scales only the reconstruction contribution:

```text
gradient(total)
  = a * gradient(reconstruction)
  + b * gradient(adversarial)
```

This changes both the magnitude and direction of the combined gradient. It is
not equivalent to scaling the complete gradient unless all loss components
are scaled by the same constant.

### Interaction with norm clipping

If the complete gradient of a group is clipped to maximum norm `M`, then:

```text
clipped_gradient
  = gradient * min(1, M / norm(gradient))
```

When the group is already above `M`, multiplying the complete loss by a
positive constant changes the raw norm but usually produces the same clipped
direction and the same final norm `M`. The total loss scale is therefore
largely cancelled by clipping.

Scaling only one loss component can still matter because it changes the
direction formed by the mixture of objectives before clipping.

This is particularly relevant for:

- latent flow, whose weight is `100` and whose module group is almost always
  clipped;
- waveform reconstruction, whose weight is `45` and whose generator group is
  always clipped;
- KL, which can redirect the audio-encoder gradient even if the encoder's final
  clipped norm remains `10`.

### Interaction with AdamW

Scaling a gradient is not generally equivalent to scaling the final AdamW
parameter update.

AdamW tracks first and second moments:

```text
update ≈ first_moment / sqrt(second_moment)
```

If every gradient were multiplied by one constant consistently, much of that
scale would cancel between the moments. Changing clipping coefficients over
time still changes moment history and optimization behavior, but the clipping
coefficient must not be interpreted as a literal learning-rate multiplier.

For example, a raw gradient clipped with coefficient `0.01` is multiplied by
`0.01` before AdamW, but this does not guarantee that the resulting parameter
update is exactly 100 times smaller.

### AMP loss scaling

FP16 `GradScaler` also multiplies the loss before backward, but it unscales the
gradients before clipping and AdamW:

```text
scaled loss -> scaled gradient -> unscale -> original finite gradient
```

For finite values, AMP loss scaling is intended to be numerically transparent.
It is not a loss weight and should not change the mathematical update.

Its purpose is to keep small FP16 backward values representable. If overflow
is detected, the optimizer step is skipped and the scale is reduced.

## FP16 numerical failures

The good run used BF16. The inspected current run uses FP16.

In the current run:

- the generator AMP scale fell from `16` to `1`;
- several optimizer-gradient overflows occurred;
- several generator forward/loss computations became non-finite;
- some failures happened after `generator_complete`, identifying optimizer
  gradient failure;
- others jumped directly from `generator_backward` to `ready`, identifying a
  forward or loss-metric failure.

This is a stronger numerical warning than the clipping rate, because the good
BF16 run did not require the current skip behavior.

## Reconstruction loss precision

The current working-tree version of `losses/acoustic.py` removed the explicit
FP32 autocast-disabled wrapper around the multi-resolution reconstruction
loss.

Mixed precision does not require every operation to execute in FP16. Spectral
transforms, large reductions, divisions, logarithms, and small denominators
are common reasons to retain FP32 locally.

The reconstruction loss performs:

- multiple mel-spectral transforms;
- large absolute-error sums;
- target-magnitude sums or norms;
- division by those target values.

Restoring an explicit FP32 numerical island would still allow the model
forward and backward to use mixed precision.

## Non-finite skip handling hides failures

The current loop catches `FloatingPointError` during:

- discriminator loss computation;
- generator loss computation;
- optimizer preparation/step.

It then discards accumulated gradients and continues.

Problems:

- the exception message and failing metric/group are not logged;
- `skipped_steps` represents consecutive skips before the next success, not a
  cumulative total;
- forward/loss failures and optimizer-gradient failures look nearly identical
  in MLflow;
- the failed batch is consumed and not retried;
- both optimizers' accumulated gradients are discarded when either side fails.

The generator scaler is reduced for detected backward-gradient overflow.
Lowering the scaler cannot repair a non-finite forward calculation because AMP
loss scaling affects backward, not the forward values.

Skipping can be a valid production policy, but it must retain the exact failure
reason and cumulative counts. Otherwise it converts a crash into silent
training corruption or an invisible skip wall.

## Loss reduction and native scale differences

The objectives do not naturally have comparable gradient scales:

- latent flow is mean squared error per valid latent scalar;
- duration flow is negative log likelihood normalized per valid token;
- KL sums latent dimensions and then averages valid frames;
- reconstruction is relative L1 error normalized by target mel magnitude;
- GAN losses sum across critics;
- feature matching sums across critic feature maps and multiplies by two;
- embedding objectives normalize embeddings and use cosine-like geometry.

Using one absolute value of `10` for all module families ignores these scale
differences.

## Other gradient-changing operations

These are intentional and are not the same as optimizer clipping:

- AdamW moment normalization and weight decay;
- style-speaker gradient reversal scaled by `-0.1`;
- detached generated audio during discriminator training;
- temporarily frozen discriminator parameters during generator loss;
- EMA latent-flow shortcut targets computed under `no_grad`;
- conditional posterior targets computed under `no_grad`;
- detached hard alignments;
- frozen text encoder;
- posterior `log_scale` clamp;
- unit-normalized style and voice embeddings.

The following primarily normalize activations or parameterization and should
not be removed as part of gradient-clipping cleanup:

- weight normalization;
- discriminator spectral normalization where configured;
- instance and layer normalization;
- residual `1 / sqrt(2)` scaling;
- masked statistical normalization.

## Recommended future cleanup

No cleanup has been applied yet.

A controlled improvement should:

1. Use BF16, or retain explicit FP32 numerical islands for sensitive spectral
   losses when testing FP16.
2. Keep unscaled non-finite gradient detection.
3. Keep module groups independent.
4. Replace norm `10` with high per-group anomaly thresholds that affect only
   approximately `0.1%` to `1%` of finite steps.
5. Log raw norm, active threshold, coefficient, exact non-finite stage,
   failing metric/group, and cumulative skip count.
6. Compare from step zero against the unchanged policy with the same seed,
   data ordering, precision, architecture, losses, and schedules.
7. Avoid adaptive clipping heuristics initially. Explicit per-group thresholds
   are easier to inspect and reproduce.

For reference, the good run's first-3,750-step 99.9th-percentile norms were
approximately:

- audio encoder: `117`;
- decoder: `90`;
- generator: `361`;
- discriminator: `144`.

The current run had rare generator spikes around `3,459` and `19,982`, while
its ordinary generator distribution remained close to the good run. This is
the type of separation an anomaly-only cap should preserve: normal gradients
pass unchanged, while genuinely exceptional spikes are bounded.
