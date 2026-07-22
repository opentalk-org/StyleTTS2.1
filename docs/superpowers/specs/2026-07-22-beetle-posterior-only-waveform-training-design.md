# Beetle Posterior-Only Waveform Training Design

## Goal

Train Beetle's acoustic decoder, waveform generator, and discriminators only from
audio-encoder posterior latents. Train the conditional model through its text,
alignment, duration, embedding, flow-matching, shortcut, and latent consistency
objectives without decoding a conditional endpoint during training.

Validation must still synthesize the conditional endpoint through the complete
decoder and generator so `step_x/full` contains audible full-pipeline output.

## Training Data Flow

The discriminator update uses one posterior reconstruction per real waveform:

```text
audio -> posterior encoder -> feature prediction -> decoder -> generator -> fake
real + fake -> discriminator loss
```

It does not build conditional synthesis inputs or evaluate the latent-flow model.

The generator update has two loss paths that meet only through their shared
optimizer:

```text
audio -> posterior encoder -> decoder/generator -> acoustic and GAN losses
text + audio -> conditional model -> latent and conditioning losses
```

The conditional path stops at latent-space objectives. It does not call
`integrate_latent_flow(..., steps=1)` and receives no reconstruction,
adversarial, feature-matching, F0, or noise-feature gradient.

## Losses and Metrics

Posterior acoustic training retains encoder KL, F0, noise-feature,
reconstruction, adversarial, and feature-matching losses. Reconstruction weight
changes from `90` to `45`, preserving the posterior branch's effective
coefficient from the previous two-branch average.

Conditional duration, alignment, embedding, latent-flow, shortcut, statistics,
speaker-adversarial, and style-reencoding losses remain unchanged. The
`posterior_reconstruction` metric remains; `conditional_reconstruction` is
removed.

## Validation

Validation continues to produce two complete reports:

- `step_x/audio`: posterior reconstruction from the audio encoder.
- `step_x/full`: conditional synthesis from text and noise through latent flow,
  feature prediction, decoder, and generator.

Aggregate acoustic validation losses follow the training objective and use only
the posterior waveform and posterior acoustic features. Conditional latent and
conditioning losses remain in the report. Full-pipeline audio is generated for
audible evaluation and artifacts, not included in aggregate acoustic losses.

## Code Boundaries

Replace the joint training synthesis helper with a posterior-only synthesis
helper. Remove the training-only conditional synthesis input type and builder
method when they have no remaining callers. Validation retains its independent
conditional synthesis implementation.

The active training process is not restarted. These changes affect only a later
training launch.

## Verification

Use temporary tests to establish that posterior training synthesis does not call
latent-flow integration and that training metrics contain no conditional
reconstruction. Verify that conditional validation still generates the full
waveform report. Run Beetle Python compilation through the Nix development
environment, then remove temporary tests.
