# Duration Normalizing Flow Convention

This note locks the Beetle duration likelihood before implementation. The
source commit is Piper `73c04d81d5590ecc46e522de3601ce7fb29fc2be`, especially
`src/python/piper_train/vits/models.py:14-117` and
`src/python/piper_train/vits/modules.py:371-527`. VITS arXiv 2106.06103,
section 2.2.1, is the primary mathematical source.

## Tensor shapes and mask

- Text condition `c`: `[B, C, T]`.
- Positive integer duration `w`: `[B, 1, T]`.
- Valid-token mask `m`: `[B, 1, T]`, with values exactly zero or one.
- Augmented variables `z`, `e_q`: `[B, 2, T]`.
- All transforms are identity outside `m`; all log determinants sum only valid
  elements and return `[B]`.

The implementation reports the mean negative log likelihood per valid token.
A sample with no valid token is invalid input rather than a silently skipped
loss.

## Forward direction: durations to base density

Piper uses variational dequantization because durations are discrete. A
posterior flow conditioned on `c` and an encoding of `w` maps
`e_q ~ Normal(0, I)` to `(z_u, z_1)`. Define

```text
u = sigmoid(z_u)
z_0 = w - u
log q(u,z_1 | w,c)
  = log Normal(e_q; 0,I) - log|det J_posterior|
    - log|det sigmoid(z_u)|
```

The main forward direction first applies `log(z_0)`, then an elementwise
affine transform and alternating conditional rational-quadratic spline
couplings and channel flips. It maps `(z_0, z_1)` to standard-normal `z`.
Each forward transform returns

```text
logdet = log |det(d output / d input)|
```

so the log-determinant sign in the main negative log likelihood is negative:

```text
nll = -log p(w | c)
    <= 0.5 * sum_valid(log(2*pi) + z^2)
       - sum(main forward logdet)
       + log q(u,z_1 | w,c)
```

This is the exact `nll + logq` convention in Piper `models.py:91-107`. Beetle
does not replace it with MSE on log duration.

## Reverse direction: base density to log duration

For inference, draw `z ~ Normal(0,I) * noise_scale`, traverse the main flows in
reverse order, and return channel `z_0` as sampled `log(w)`. The redundant
terminal variational-flow pairing is removed exactly as in Piper
`models.py:108-117`. Convert to usable durations only at the caller boundary:

```text
w = ceil(exp(log_w)) * mask
```

The caller applies explicit configured clamps before expanding alignments.

## Required implementation checks

1. Every transform reconstructs valid values after forward then reverse.
2. Forward and reverse analytic log determinants cancel.
3. A tiny finite-difference Jacobian agrees with the analytic determinant.
4. Padding does not affect likelihood or sampling.
5. Base density, posterior density, sigmoid Jacobian, and main Jacobian all
   participate in the negative log likelihood.
