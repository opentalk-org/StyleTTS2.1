# Beetle Signal Checkpoint Design

## Goal

Make standalone Beetle training save a normal recoverable checkpoint before it
exits after SIGINT or SIGTERM.

## Behavior

The signal handler continues to request cancellation rather than performing
I/O directly. Training observes the request only at an exact optimizer-step
boundary, after accumulated microsteps have been consumed, the optimizer has
stepped, and reporting has completed. It then flushes reporting, saves a
checkpoint with `microstep=0` and `phase=READY`, publishes the checkpoint
artifact, and exits cleanly.

If a signal arrives during partial gradient accumulation, training completes
the remaining microsteps and the associated optimizer step before observing
the cancellation request. This preserves the checkpoint payload invariant that
partial accumulation is never serialized.

SIGINT and SIGTERM retain identical behavior because both are deliberate
requests for a recoverable shutdown.

## Scope

The change is confined to Beetle's training loop. It does not checkpoint from
inside a signal handler, change periodic checkpoint scheduling, change the
checkpoint format, or affect exceptions other than `CancellationRequested`.
Signals received before model and pipeline construction cannot save a training
checkpoint because no complete runtime state exists yet.

## Verification

Temporary tests exercise cancellation at a ready boundary and during partial
gradient accumulation. They assert that checkpoint saving occurs exactly once,
the serialized state is ready with zero microsteps, and training exits only
after the optimizer boundary. Temporary tests are removed after verification
in accordance with repository policy.
