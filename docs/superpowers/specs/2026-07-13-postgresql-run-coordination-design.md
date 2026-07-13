# PostgreSQL Run Coordination Design

## Goal

Replace NATS with PostgreSQL as the durable coordination and observability system for backend and runner processes. Persist bounded replacement state rather than per-item event deltas, and coalesce writes so database traffic scales with active runs and flush intervals rather than processed items.

## Architecture

PostgreSQL tables are authoritative. `LISTEN/NOTIFY` carries identifiers only and wakes processes after committed state changes; periodic polling remains mandatory because notifications are not durable. Backend and runner behavior must remain correct if every notification is lost or duplicated.

The backend creates jobs and changes desired state. Runners claim queued jobs, reconcile desired state, execute graphs, and write observed state. Both active and historical APIs read the same persisted job snapshot. No in-memory NATS-derived snapshot has separate semantics.

## Job Dispatch and Leases

Each job stores:

- desired state: `running` or `stopped`
- observed lifecycle state: `queued`, `running`, `stopping`, `succeeded`, `failed`, or `stopped`
- optional target runner and claimed runner
- lease expiry and last lease renewal
- graph request and latest run snapshot
- lifecycle timestamps and terminal error

A runner claims eligible queued jobs with `SELECT ... FOR UPDATE SKIP LOCKED`. A claim changes observed state and lease fields in the same transaction. A runner renews leases while executing. Expired non-terminal claims return to the queue only when their runner heartbeat is stale, preventing a slow database operation from causing duplicate execution.

Stop requests replace desired state with `stopped`. Active runners poll desired state and also listen for a job-state notification. Cancellation is checked between scheduler batches and during existing long-running node cancellation points.

## Desired Node State

Node lifecycle control uses a per-run, per-node state row containing desired loaded state, observed loaded state, reconciliation error, and update timestamps. Backend requests replace desired state. The claimed runner reconciles it and replaces observed state. Duplicate requests are harmless.

## Bounded Run State

The runner maintains counters in memory and periodically replaces the complete `jobs.snapshot` JSONB value. The snapshot contains node status, aggregate input/output counts and rates, queue depth, bounded recent batch samples, and bounded errors. Packet creation, packet delivery, task enqueue, and individual log-line events are not persisted.

State persistence limits are:

- run snapshot: at most once per 500 milliseconds per active run
- node logs: at most once per second per active run
- runner heartbeat and active-run state: once per five seconds
- lease renewal: once per five seconds
- lifecycle transitions, failures, cancellation completion, and successful completion: immediate synchronous flush

Updates arriving inside a rate-limit window mark their state dirty and replace the pending value. They do not queue another database operation. A terminal flush writes the latest snapshot, logs, job lifecycle, lease release, and runner state before execution is reported finished.

## Shared Auto-Flush Boundary

Runner code submits typed state replacements to a shared flush buffer. The buffer owns dirty-state coalescing and timing but does not issue ad hoc SQL. A shared run-coordination CRUD facade accepts one typed flush payload and performs bulk upserts in one database transaction:

- job lifecycle and snapshot replacements
- runner heartbeat and active run IDs
- desired/observed node state replacements
- capped node log replacements

The facade uses PostgreSQL bulk insert/upsert statements. It emits `pg_notify` only after the affected rows have been written in the same transaction. Notification payloads contain the changed run or runner identifier, never state bodies.

## Logs

Node log handlers retain capped text state per node. The flush buffer collects all dirty node logs and bulk-upserts them once per log interval. Logs flush immediately on node failure and run termination. PostgreSQL stores only the latest capped content, truncation flag, and optional error, so log volume is bounded by node count and cap size.

The backend reads logs directly from PostgreSQL. Runner log request/response commands are removed.

## Backend Synchronization

The backend listens for committed run and runner notifications. On wake-up it reloads authoritative rows and broadcasts updated API/WebSocket state. A periodic updated-at scan repairs missed notifications and restart gaps. Active and historical views therefore share identical serialization and counters.

Backend start, stop, and node lifecycle endpoints update desired database state through CRUD functions. They do not wait for an ephemeral acknowledgement; responses expose desired and observed state so pending reconciliation is visible.

## Runner Presence

Runner registration and presence are persisted in the runners table using replacement heartbeat state: identity, host information, last-seen timestamp, process identifier, capabilities, and active run IDs. Online/stale status is derived from the timestamp. The backend has no separate in-memory heartbeat registry.

## Failure Handling

- Database unavailability pauses job claiming and retries state flushes without discarding the latest dirty replacement state.
- A terminal run is not considered durably complete by the runner until its synchronous terminal flush commits.
- Backend or runner restarts recover entirely from current PostgreSQL rows.
- Notification loss only adds polling latency.
- Duplicate claims are prevented by row locking and leases; stale leases are recovered only after heartbeat expiry.
- Snapshot and log sizes are bounded, so no run produces millions of observability rows.

## Removal Scope

After PostgreSQL coordination is wired end to end, remove the backend NATS bus, runner JetStream consumers/publishers, shared JetStream helpers and message schemas, NATS dependency, NATS process startup, NATS ports, and deployment topology references. PostgreSQL remains required for every runner.

## Verification

Verification covers one normal run, stop reconciliation, targeted runner claiming, backend and runner restart recovery, stale lease recovery, node desired-state reconciliation, live capped logs, and a 100,000-item metadata import. The large import must produce database writes proportional to flush intervals and batches, show the same counters before and after completion, and reach terminal state without draining an item-event backlog.
