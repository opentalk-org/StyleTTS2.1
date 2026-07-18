# Beetle Posterior Encoder Design

## Goal

Reduce the temporal field of view and computation of Beetle's audio posterior
encoder while preserving its approved 40 Hz latent interface.

## Architecture

The encoder consumes 80-bin mel features at 80 Hz. A kernel-4, stride-2
convolution projects the mel features to 192 channels and establishes the
40 Hz latent clock. The projected sequence then passes through 16
Piper/VITS-style gated residual layers. Every layer uses a kernel size of five,
dilation one, 192 hidden channels, weight-normalized convolutions, and the
configured dropout.

The first 15 layers split their one-by-one projection into residual and skip
paths. The final layer contributes only to the skip path. Accumulated skip
features pass through a one-by-one projection that produces the 192-channel
posterior mean and log scale. Sampling, masks, log-scale bounds, and the
`AudioPosterior` interface remain unchanged.

## Geometry and complexity

The theoretical temporal receptive field is:

```text
4 + 2 * (5 - 1) * 16 = 132 mel frames
132 / 80 Hz = 1.65 seconds
```

The default encoder has 7,200,960 trainable parameters when weight-normalization
parameters are counted. Its convolution work for ten seconds of audio is about
5.74 GFLOPs under the project's PyTorch operation-counting convention.

The encoder continues to return `[B,192,T/2]`. The aligner, latent flow,
FeatureLinear, Decoder, Generator, and context/style/voice consumers therefore
require no interface or rate changes.

## Configuration

Posterior configuration explicitly declares `layer_count: 16`. The obsolete
dilation cycle and cycle count are removed. Default hidden width changes from
256 to 192. Kernel size five, dropout, stride, posterior bounds, and latent
width remain explicit configuration values.

## Ownership

The gated posterior stack lives inside Beetle's model family. It does not
import implementation code from Piper or another runner node family. Public
module and variable names describe stable model responsibilities rather than
the source project.

## Verification

A temporary focused check must fail against the old configuration and then
confirm the new layer geometry, parameter count, receptive field, output shape,
mask behavior, and backward gradients. Project Python commands run through the
Nix development shell. Temporary checks are removed before completion.
