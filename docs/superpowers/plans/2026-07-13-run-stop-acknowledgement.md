# Run Stop Acknowledgement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve `stopping` until runner acknowledgement and guarantee that every runner-confirmed `stopped` job has no active nodes.

**Architecture:** `RunnerStateBuffer.mark_terminal` records the terminal run event before producing the database replacement, so clean acknowledgements update job and snapshot together. Expired stopping claims are finalized only from runner-invoked coordination recovery, using a typed snapshot projection shared with no frontend code.

**Tech Stack:** Python 3.12, asyncio, Pydantic 2, SQLAlchemy 2, PostgreSQL, unittest, Nix development shell

## Global Constraints

- Backend stop requests may set an executing job only to `stopping`.
- `stopping` jobs are never claimable.
- Only runner acknowledgement or runner-owned expired-lease recovery sets an executing job to `stopped`.
- A stopped snapshot contains no active node, loaded resource, running batch, current timer, or queued work.
- Use typed snapshot models; do not mutate raw JSON dictionaries.
- Add no frontend masking, legacy graph parsing, or compatibility path.
- Remove throwaway tests before completion.

---

### Task 1: Normal runner stop acknowledgement

**Files:**
- Modify: `src/runner/state_buffer.py:74-80`
- Modify: `src/shared/event_store.py:234-244`
- Test temporarily: `/tmp/test_run_stop_acknowledgement.py`

**Interfaces:**
- Consumes: `RunnerStateBuffer.event_sink(event: RunEvent)` and `RunEventStore` terminal event projection
- Produces: `mark_terminal(run_id, "stopped")` records `RunEvent(kind="run_stopped")` before releasing the claim

- [ ] **Step 1: Write the failing test**

Create a `unittest.TestCase` that registers a buffered run, records `batch_started` for node `source`, calls `mark_terminal(run_id, "stopped")`, and asserts the snapshot node is `stopped` with zero running batches and no current batch timestamp.

- [ ] **Step 2: Verify the red state**

Run: `nix develop --command python -m unittest discover -s /tmp -p 'test_run_stop_acknowledgement.py' -v`

Expected: FAIL because `mark_terminal` changes only the job state.

- [ ] **Step 3: Implement terminal event recording**

Before changing the buffered job fields, record `run_stopped` for `state == "stopped"` and `run_failed` for `state == "failed"`. Ensure stopped terminal projection also clears `loaded`. Successful runs retain the scheduler's existing `run_completed` event.

- [ ] **Step 4: Verify the green state**

Run the same unittest command. Expected: the normal acknowledgement test passes.

### Task 2: Runner-owned expired lease recovery

**Files:**
- Create: `src/shared/run_snapshots.py`
- Modify: `src/shared/db/jobs/coordination_crud.py:15-48,223-243`
- Extend temporary test: `/tmp/test_run_stop_acknowledgement.py`

**Interfaces:**
- Produces: `stopped_run_snapshot(snapshot: RunSnapshot, message: str) -> RunSnapshot`
- Consumes: the helper from `_recover_expired_claims(session, now)` for expired jobs with `desired_state == "stopped"`

- [ ] **Step 1: Extend the failing test**

Build a typed `RunSnapshot` containing an active, loaded node with a current batch timer. Assert `stopped_run_snapshot` returns a snapshot with `run_stopped` counted once, status `stopped`, `loaded=False`, `queue_size=0`, `running_batches=0`, and cleared current timing.

- [ ] **Step 2: Verify the red state**

Run the unittest command. Expected: ERROR because `shared.run_snapshots` does not exist.

- [ ] **Step 3: Implement typed terminal projection**

Create the focused helper with Pydantic `model_copy` calls. Update recovery to lock expired stopping rows, validate each stored snapshot, apply the helper, set terminal fields, and notify when either recovery or claiming changes jobs. Keep claim candidates restricted to queued/running-desired jobs.

- [ ] **Step 4: Run verification**

Run the unittest command, Python compilation for all changed modules, `git diff --check`, and a PostgreSQL query confirming the recovery and claim predicates. Expected: all commands exit zero.

- [ ] **Step 5: Clean up**

Delete the temporary test and leave unrelated worktree changes untouched. Do not create a branch, worktree, commit, or legacy migration.
