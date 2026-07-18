# Beetle Adversarial Training Stages

## Stage contract

Stage 1 trains the posterior latent-to-audio path and both approved StyleTTS
discriminator families. Each microstep performs one discriminator backward pass
against detached posterior reconstruction, then one generator backward pass
containing KL, F0, N, reconstruction, generator-adversarial, and feature-matching
terms. Discriminator, adversarial-generator, and feature-matching weights warm up
independently from configuration.

Stage 2 freezes the Stage 1 audio path and trains no discriminator. Stage 3
trains both discriminators again while jointly fine-tuning the posterior and
text-conditioned audio paths with the Stage 2 objectives.

## Optimizers and resumability

Stages 1 and 3 each own separate generator and discriminator optimizers,
schedules, AMP scalers, gradient metrics, checkpoint states, and partial-step
gradients. A completed Stage 1 checkpoint includes discriminator weights, and
Stage 3 initializes its discriminators from that checkpoint. Stage 2 has only
its generator optimizer.

## Reporting and validation

Stage 1 training reports raw `discriminator`, weighted `discriminator_total`,
raw `generator_adversarial`, raw `feature_matching`, the four acoustic losses,
and weighted `generator_total`. MLflow receives them under `train/`.

Stage 1 validation evaluates the same loss set without optimizer updates and
publishes aggregate `validation/` metrics. Stage 3 keeps its existing posterior
and conditional averaging. No per-sample metrics are sent to MLflow.

## Compatibility

The correction does not mutate or restart a running process. A Stage 1 process
started with the non-adversarial implementation must be restarted from the
beginning with a configuration containing the required discriminator optimizer;
its checkpoint does not contain trained discriminator state.
