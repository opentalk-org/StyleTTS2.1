# Beetle Single-Stage Window-Prefetch Design

## Goal

Replace Beetle's three sequential training stages with one end-to-end training
run. The acoustic path, conditioning models, duration and latent flows, and GAN
train together from the first optimizer step. Training resumes only from the
last completed optimizer step.

The data path remains padded rather than packed. Acoustic reconstruction and
adversarial work always use a 0.8-second tensor; shorter recordings retain all
available audio and receive zero padding. Full-sequence objectives continue to
use the recording's valid masks. Existing loss reductions are retained, so this
change does not introduce equal-per-recording weighting.

## Single training lifecycle

One trainer owns all trainable Stage 1 and Stage 2 modules, the latent-flow EMA,
generator optimizer, discriminator optimizer, schedules, validation, and
checkpoint state. It implements the current joint Stage 3 objective without a
Stage 1 or Stage 2 transition and initializes every model directly for the one
run.

For the rapid 2,000-step test run, GAN loss weights use a 1,000-step warmup.
The scheduled-weight schema still permits disabling or delaying each term by
setting its start step to zero or a value beyond the run. Pretrained
fixed feature extractors remain frozen; "single stage" removes training-stage
freezing and handoff, not the architectural role of fixed reference models.

Configuration exposes one `training` section rather than `stage1`, `stage2`,
and `stage3`. Stage-specific entry points and checkpoint payload branches are
removed. Metrics and validation use a single training namespace.

## Acoustic segment geometry

Every acoustic segment contains 19,200 waveform samples at 24 kHz, 64 acoustic
frames at hop 300, and 32 latent frames after posterior downsampling.

Segment starts are chosen from valid frame lengths and aligned to latent-frame
boundaries. Recordings shorter than 64 acoustic frames start at frame zero and
use the existing right-side zero padding. Their ordinary frame and sample masks
continue to identify the real portion, while the fixed tensor geometry keeps
generator and discriminator shapes stable.

## Double-buffered prefetch windows

`data.prefetch.window_size` counts batches per buffer. Each buffer owns
`window_size * batch_size` local examples; two buffers allow the active window
to feed training while the standby window is planned, fetched, decoded, and
collated.

For every window, the deterministic planner draws the same uniformly shuffled
examples it would otherwise draw. Known valid durations are used to sort the
window before expensive decoding. Adjacent examples form fixed-size batches,
then those completed batches are deterministically shuffled. Thus each window
contains duration-homogeneous batches but can execute in any length order, such
as long, shortest, then medium. Every example is consumed exactly once.

For distributed execution, duration grouping operates on each deterministic
global batch window before rank sharding so all ranks execute similar shapes.
Voice and style auxiliary groups remain complete and are assigned once per
emitted batch.

The current decoded-byte and prepared-batch bounds remain authoritative. A
window configuration that cannot coexist with those bounds fails during setup
instead of partially filling a buffer. A window size of one preserves ordinary
fixed-batch behavior.

## Exact recovery

A checkpoint represents the last completed optimizer step only. It stores the
model, optimizer, scaler, EMA, scheduler, loop, RNG, and committed planner state
associated with that boundary. It does not save decoded or prefetched audio.

Planning and prefetch may run ahead, but only batches included in a completed
optimizer step advance committed data state. Resume restores that state and
deterministically regenerates uncommitted windows. Partial-step gradients,
stage-transition payloads, and inferred off-by-one recovery paths are removed.
Checkpoint discovery selects the greatest valid completed step and fails
clearly on malformed or internally inconsistent state.

## Verification

Temporary tests and real graph runs must establish that:

1. configuration and runtime expose one training lifecycle;
2. all intended modules receive gradients from the first optimizer step;
3. acoustic tensors are always 19,200 samples and short recordings cannot
   produce an all-padding segment;
4. two deterministic windows overlap preparation and consumption;
5. examples are grouped by duration while completed batch order is shuffled;
6. no example is duplicated or omitted across a window or distributed shard;
7. checkpoints resume at exactly the last completed optimizer step and
   regenerate uncommitted prefetch work;
8. the existing smoke workflow completes through the real runner graph.
