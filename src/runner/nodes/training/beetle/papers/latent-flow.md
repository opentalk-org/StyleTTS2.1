# Conditional Latent Flow and Shortcut Convention

This note defines the deliberate merge used by Beetle. Primary sources are
Flow Matching arXiv 2210.02747 equations 9 and 14, Diffusion Forcing arXiv
2407.01392 section B.2, and Shortcut Models arXiv 2410.12557 equations 3-5
and algorithm 1. The local StyleTTS2 source commit is
`5cedc71c333f8d8b8551ca59378bdcc7af4c9529`; it supplies conditioning and TTS
context but not this objective.

## Tensor contract

Latents, noise, and velocity use `[B, C, T]`. Token mask, noise level `t`, and
requested step size `d` use `[B, 1, T]`. Conditions are projected token
sequences with the same `T`. Invalid positions are exact zeros at every
construction and reduction boundary.

## Probability path and analytic velocity

Use the straight conditional optimal-transport probability path from Gaussian
noise `x_0` to AudioEncoder latent `x_1`:

```text
x_t = (1 - t) * x_0 + t * x_1
u_t = d(x_t)/dt = x_1 - x_0
x_0 ~ Normal(0, I), t[b,1,j] ~ Uniform(0,1)
```

The base conditional flow-matching loss is masked MSE between the model output
`v_theta(x_t,t,d=0,conditions)` and analytic velocity `u_t`. Independent
per-token `t` is the adopted Diffusion Forcing property. Beetle does not claim
to reproduce its autoregressive full-sequence objective; it applies the
independent-noise principle to a masked bidirectional temporal CNN.

## Dyadic shortcut target

The public `d` is the full requested step. For nonzero dyadic `d`, constrain
`t + d <= 1` per valid token and build a target using EMA parameters:

```text
v_a = ema(x_t, t, d/2, conditions)
x_mid = x_t + (d/2) * v_a
v_b = ema(x_mid, t + d/2, d/2, conditions)
v_target = stop_gradient((v_a + v_b) / 2)
loss_shortcut = masked_mse(online(x_t,t,d,conditions), v_target)
```

At the smallest represented half-step, the EMA queries use `d=0`, matching
Shortcut Models algorithm 1. `d=0` always uses the empirical analytic velocity
base case. Base and shortcut cases mix within a batch. EMA targets are
evaluated before the online optimizer update; EMA updates exactly once after
that update.

## Per-token time and step sampling

Choose a configured smallest interval `1/M`, where `M` is a power of two.
For each valid token, sample a dyadic full step from
`{2/M, 4/M, ..., 1}` for shortcut cases, then sample `t` from multiples of
`d` in `[0, 1-d]`. Base cases independently sample continuous `t` and set
`d=0`. Stateless seeds include stage, cycle, batch, sample, token, and view so
resume recreates every value.

## One-step and multi-step sampling

Generation begins at masked standard Gaussian noise. For `S` steps, use
`d=1/S` and the Euler/shortcut update:

```text
x <- x + d * v_theta(x, t, d, conditions)
t <- t + d
```

One-step generation is `S=1`, `t=0`, `d=1`. Multi-step generation requires
`S` compatible with configured dyadic training steps. The final mask is
applied after every update.

## Stop-gradient and verification requirements

- EMA bootstrap calls run without autograd and their combined target is
  detached.
- Gradients enter only the online prediction for a shortcut loss.
- Hand-constructed path/velocity examples must match the equations token by
  token.
- One full shortcut target must equal the average direction of its two EMA
  half steps.
- One-step and multi-step integrators must use the same step-conditioned model
  and direction convention.
