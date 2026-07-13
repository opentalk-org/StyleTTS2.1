# Run Stop Acknowledgement Design

## State machine

An executing job transitions from `running` to `stopping` when the backend records a stop request. The backend does not set an executing job to `stopped`. While the request is pending, node snapshots may continue to report `running` because the runner may still be executing or frozen.

A responsive runner acknowledges the request by cancelling execution, recording `run_stopped`, and flushing the job state and terminal node snapshot together. A stopped snapshot has no active node statuses, running batches, current batch timer, queue depth, or loaded resource state.

If a runner disappears, the job remains `stopping` until its lease expires. On startup or polling, a runner performs expired-lease recovery. A job whose desired state is `stopped` is never reclaimed; the recovering runner records the equivalent terminal snapshot and changes the job to `stopped` in the same database transaction.

Queued jobs may become `stopped` directly because no runner owns or executes them.

## Ownership and invariants

- Backend actions request state changes; they do not acknowledge active-run termination.
- Runner execution acknowledges normal cancellation.
- Runner-invoked lease recovery acknowledges that an expired owner can no longer be trusted to execute the job.
- Claim selection remains limited to `state=queued` and `desired_state=running`, so `stopping` jobs cannot restart.
- Every persisted `stopped` snapshot has all active nodes projected to `stopped`, `running_batches=0`, `queue_size=0`, `loaded=false`, and no current batch timing.
- Recovery uses typed `RunSnapshot` and `NodeRunSnapshot` models rather than raw JSON mutation.
- No frontend override, legacy graph parsing, or compatibility behavior is introduced.

## Verification

Throwaway tests will first demonstrate that normal terminal marking leaves an active node running and that recovery projection leaves active fields intact. After implementation, the same tests must pass. Python compilation and database-facing smoke checks will run through the Nix development shell, and temporary tests will be removed before completion.
