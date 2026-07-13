# PostgreSQL Run Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace NATS workflow coordination with rate-limited, state-based PostgreSQL dispatch, observability, logs, runner presence, and live backend synchronization.

**Architecture:** PostgreSQL rows are authoritative and are replaced idempotently. Runners claim jobs with row locks, coalesce in-memory run snapshots and capped logs, then bulk-flush them through one shared CRUD transaction. PostgreSQL notifications wake pollers but never carry authoritative data.

**Tech Stack:** Python 3.12, SQLAlchemy 2, PostgreSQL JSONB and `LISTEN/NOTIFY`, FastAPI, asyncio, Pydantic.

## Global Constraints

- Run-state flushes occur at most once per 500 milliseconds per active run.
- Capped node logs flush at most once per second per active run.
- Heartbeats and leases update every five seconds.
- Terminal lifecycle and failures flush synchronously.
- Persist replacement state and bounded samples; never persist per-item events.
- All database access goes through `src/shared/db/**/crud.py` and transaction boundaries stay there.
- Keep files under 300 lines and folders under 16 files.
- Work in the current checkout; do not create a worktree.
- Do not add committed test files, matching the user's standing instruction.

---

### Task 1: PostgreSQL coordination schema and CRUD

**Files:**
- Create: `migrations/versions/20260713_1400_d5e6f7a8b9c0_postgres_run_coordination.py`
- Modify: `src/shared/db/jobs/models.py`
- Modify: `src/shared/db/jobs/schemas.py`
- Modify: `src/shared/db/jobs/crud.py`
- Modify: `src/shared/db/runners/models.py`
- Modify: `src/shared/db/runners/schemas.py`
- Modify: `src/shared/db/runners/crud.py`
- Create: `src/shared/db/jobs/coordination_crud.py`

**Interfaces:**
- Produces: `claim_jobs(runner_id: str, limit: int) -> list[ClaimedJob]`.
- Produces: desired-state mutations for stop and node loaded state.
- Produces: `flush_runner_state(payload: RunnerStateFlush) -> None`, one transaction containing job snapshots, logs, node states, runner heartbeat, leases, and `pg_notify` calls.
- Produces: updated-row scans used by backend and runner polling loops.

- [ ] Add job target/claim/desired-state/lease columns, persistent runner heartbeat state, and a `run_node_states` table with desired and observed loaded state.
- [ ] Add typed Pydantic payloads for claimed jobs, snapshot replacements, node-state replacements, log replacements, and runner flushes.
- [ ] Implement atomic job claiming with `FOR UPDATE SKIP LOCKED`, stale-lease recovery, desired-state reads, and bulk PostgreSQL upserts.
- [ ] Emit run/runner notifications after state mutation in the same transaction.
- [ ] Run `nix develop --command alembic upgrade head` and inspect the resulting schema.

### Task 2: Rate-limited runner state buffer

**Files:**
- Create: `src/runner/state_buffer.py`
- Modify: `src/runner/node_logs.py`
- Modify: `src/shared/event_store.py`
- Modify: `src/runflow/runtime/output_router.py`
- Modify: `src/runflow/runtime/scheduler.py`

**Interfaces:**
- Consumes: `flush_runner_state(payload: RunnerStateFlush) -> None` from Task 1.
- Produces: `RunnerStateBuffer.record_event`, `mark_logs_dirty`, `flush_due`, and `flush_terminal`.

- [ ] Make `RunEventStore` convert scheduler events into bounded current state while counting completed items from batch sizes.
- [ ] Remove packet-created, packet-delivered, and task-enqueued persistence from normal routing so a 100,000-item batch stream produces only batch-level state changes.
- [ ] Let capped log handlers mark nodes dirty without performing database writes from logging threads.
- [ ] Implement latest-value coalescing with 500 ms snapshot, one-second log, and five-second heartbeat/lease deadlines.
- [ ] Make terminal flush collect all dirty node logs and the latest run snapshot into one CRUD call.

### Task 3: PostgreSQL runner worker

**Files:**
- Create: `src/runner/job_poller.py`
- Create: `src/runner/run_execution.py`
- Rewrite: `src/runner/worker.py`
- Rewrite: `src/runner/heartbeat.py`
- Modify: `src/runner/cli.py`

**Interfaces:**
- Consumes: job claiming, desired state, node desired state, and flush interfaces from Tasks 1-2.
- Produces: a runner loop with no NATS dependency.

- [ ] Poll and claim targeted or unassigned queued jobs, using PostgreSQL notifications only to reduce polling latency.
- [ ] Start claimed graphs, attach the state buffer as event sink, and immediately persist the running transition.
- [ ] Reconcile stop and node desired states on notification or polling fallback.
- [ ] Renew heartbeat/leases every five seconds through the shared batch flush.
- [ ] Persist stopped, succeeded, build-failed, and execution-failed terminal states synchronously before releasing local run state.
- [ ] Remove runner-side NATS consumers, publishers, command schemas, and CLI NATS arguments.

### Task 4: PostgreSQL backend synchronization

**Files:**
- Create: `src/backend/db_watcher.py`
- Create: `src/backend/run_records.py`
- Rewrite: `src/backend/service.py`
- Modify: `src/backend/api.py`
- Modify: `src/backend/runners/service.py`
- Modify: `src/backend/runners/api.py`
- Remove: `src/backend/nats_bus.py`
- Remove: `src/backend/jobs/persistence.py`

**Interfaces:**
- Consumes: authoritative job, runner, node state, and log CRUD from Task 1.
- Preserves: existing FastAPI response schemas and WebSocket message shapes.

- [ ] Make start insert a queued job row, stop replace desired state, and node lifecycle endpoints replace desired loaded state.
- [ ] Read status, snapshot, graph, errors, and logs directly from PostgreSQL for active and historical runs.
- [ ] Add a notification listener with updated-at polling fallback that reloads changed rows and broadcasts existing WebSocket messages.
- [ ] Derive runner online/stale state from persisted heartbeat timestamps and active run IDs.
- [ ] Remove backend in-memory event replay and NATS command bus ownership.

### Task 5: Remove NATS runtime and deployment dependencies

**Files:**
- Remove: `src/shared/jetstream.py`
- Modify: `src/shared/schemas.py`
- Modify: `src/shared/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `flake.nix`
- Modify: `nix/runflow-dev.sh`
- Modify: `nix/entrypoint.sh`
- Modify: `nix/runner-entrypoint.sh`
- Modify: `nix/runner-launch.sh`

**Interfaces:**
- Consumes: completed PostgreSQL backend and runner paths.
- Produces: a stack with PostgreSQL, PgBouncer, storage, backend, and runners but no NATS process or port.

- [ ] Delete obsolete NATS messages, helpers, dependency, startup processes, command-line flags, environment variables, ports, and tailnet service configuration.
- [ ] Update the lock through `nix develop --command uv lock`.
- [ ] Run Python compile/import checks through `nix develop --command`.
- [ ] Start the shared development session and confirm backend and runner both register and exchange job state through PostgreSQL.
- [ ] Submit one small graph through `POST /graphs/runs`, inspect its status/logs with the CLI, and confirm terminal and historical snapshots are identical.
