# Posterior-Coupled Noise for Latent Flow

The audio encoder produces a Gaussian posterior latent:

$$
z = \mu + \sigma\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Because training already has $z$, $\mu$, and $\log\sigma$, we can recover the exact noise used to sample that latent:

$$
\epsilon = (z - \mu)e^{-\log\sigma}
$$

In code, this is:

```python
source_noise = (latent - mean) * torch.exp(-log_scale)
```

## Previous independent coupling

Previously, latent-flow training generated a separate random source $\eta$:

$$
x_0 = \eta, \qquad x_1 = z = \mu + \sigma\epsilon
$$

The target velocity was therefore:

$$
v = x_1 - x_0 = \mu + \sigma\epsilon - \eta
$$

Here, $\eta$ and $\epsilon$ are independent. The model had to connect one random identity to an unrelated random target identity. This added avoidable variance to every training example.

Independent coupling is not mathematically invalid for flow matching, but it creates a harder and noisier transport problem.

## Posterior-coupled source noise

The flow source now uses the posterior's own $\epsilon$:

$$
x_0 = \epsilon, \qquad x_1 = \mu + \sigma\epsilon
$$

The target velocity becomes:

$$
v = \mu + (\sigma - 1)\epsilon
$$

The source and target now represent the same stochastic sample. The flow only needs to reshape a standard-normal sample according to the conditional posterior instead of translating between two independently sampled points.

## Inference

At inference, the source is still sampled as:

$$
x_0 \sim \mathcal{N}(0, I)
$$

The posterior's recovered $\epsilon$ is trained to follow this same distribution, so inference uses the same source distribution as training. Ground-truth audio or posterior values are not required at inference.

## Important caveat

The reduction in `latent_flow` from approximately `1.59` to `0.55` partly reflects an easier, lower-variance transport target. It does not by itself prove that generated audio improved.

The loss can improve without equivalent inference gains when:

- the posterior is not sufficiently close to a standard normal distribution;
- the inference conditioning cannot determine the posterior distribution;
- the model learns the coupled training paths but does not generalize to sampled noise;
- other predicted inputs, such as duration, F0, or noise, remain inaccurate.

Generated-audio evaluation after sufficient training is therefore the decisive test.
