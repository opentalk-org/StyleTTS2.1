# Beetle Sequence Flow Sampling Design

## Goal

Make Beetle latent-flow training match synchronous inference by sampling one
flow time and step size per batch item instead of independently per temporal
position, then start a fresh training run initialized from the step-8000
acoustic reconstruction stack.

## Sampling change

`sample_flow_training_case` will sample the base/shortcut choice, step index,
and start time with shape `[B, 1, 1]`. Those values will be expanded across the
valid temporal mask before constructing the interpolated state. Gaussian noise
remains independent for every latent channel and temporal position. There is no
configuration switch; this directly replaces per-token temporal sampling.

The existing requirement that a mixed base/shortcut sample contains both cases
will apply across valid batch items rather than temporal positions. A batch of
one cannot guarantee both cases and will retain its sampled case.

## Fresh initialization

Training will begin at optimizer step zero in a separate output directory. A
one-off initialization artifact will load only these model states from the
existing step-8000 checkpoint:

- `audio_encoder`
- `feature_linear`
- `decoder`
- `generator`

All conditional models, discriminators, frozen helpers, EMA state, optimizer
state, scheduler state, random state, sampler position, reporting state, and
loss-schedule state will be freshly initialized.

## Verification

A temporary test will establish the old failure and verify that every valid
position within a batch item receives the same time, step, and step index after
the change. It will also verify masks and independent latent noise. The test
will be removed before handoff, following repository policy. The restarted run
must log optimizer step zero and produce advancing training steps without a
restore mismatch.
