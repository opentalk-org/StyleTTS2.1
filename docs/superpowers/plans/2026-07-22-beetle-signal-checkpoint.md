# Beetle Signal Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save a recoverable Beetle checkpoint at the next optimizer boundary after SIGINT or SIGTERM.

**Architecture:** Signal handlers continue to set the existing cancellation flag. Training stops polling that flag during announced phases and partial accumulation, polls it after completed step work, and routes `CancellationRequested` through a checkpointing cancellation function.

**Tech Stack:** Python, PyTorch training loop protocols, pytest through the Nix development shell.

## Global Constraints

- SIGINT and SIGTERM have identical recoverable-shutdown behavior.
- Never serialize partial gradient accumulation.
- Do not perform checkpoint I/O inside a signal handler.
- Do not change periodic checkpoint scheduling or checkpoint format.
- Temporary tests must be removed before completion.

---

### Task 1: Specify cancellation boundaries and checkpointing

**Files:**
- Create temporarily: `src/runner/nodes/training/beetle/training/_temporary_test_signal_checkpoint.py`
- Modify: `src/runner/nodes/training/beetle/training/loop.py`
- Modify: `src/runner/nodes/training/beetle/training/loop_events.py`

**Interfaces:**
- Consumes: `CancellationRequested`, `save_checkpoint(...)`, and existing loop protocols.
- Produces: `cancel_run(...)` that flushes and saves, plus cancellation polling only at exact optimizer boundaries.

- [ ] **Step 1: Write failing temporary tests**

Create focused fakes and tests asserting:

```python
def test_announcements_do_not_observe_signal_cancellation() -> None:
    announce(owner, cancelling_callbacks, ready_state, (), StepTimer())
    assert owner.state == ready_state


def test_partial_accumulation_defers_signal_cancellation() -> None:
    _complete_accumulation(
        trainer_with_microstep_one_of_two,
        pipeline,
        cancelling_callbacks,
        None,
        reporting,
        lifecycle,
        StepTimer(),
    )
    assert trainer_with_microstep_one_of_two.state.phase is TrainingPhase.READY


def test_cancel_run_flushes_and_saves_ready_checkpoint() -> None:
    result = cancel_run(
        trainer,
        pipeline,
        callbacks,
        checkpoint_manager,
        reporting,
        lifecycle,
        StepTimer(),
    )
    assert result.phase is TrainingPhase.READY
    assert checkpoint_manager.saved_payload.loop.microstep == 0
```

- [ ] **Step 2: Verify the tests fail for the intended reasons**

Run `nix develop --command python -m pytest src/runner/nodes/training/beetle/training/_temporary_test_signal_checkpoint.py -q`.

Expected: announcement and partial-accumulation tests raise
`CancellationRequested`; `cancel_run` rejects the expanded argument list.

- [ ] **Step 3: Implement boundary-safe cancellation**

Change the `run_continuously` cancellation handler to call:

```python
return cancel_run(
    trainer,
    pipeline,
    callbacks,
    checkpoint_manager,
    reporting,
    lifecycle,
    timer,
)
```

Remove `callbacks.check_cancel()` from `announce(...)` and from the partial
accumulation return path. Expand `cancel_run(...)` to flush reporting, mark it
flushed, and return:

```python
return save_checkpoint(
    trainer,
    pipeline,
    callbacks,
    checkpoint_manager,
    reporting.snapshot(),
    trainer.loop_state(),
    timer,
)
```

Retain reporter failure behavior when flushing fails.

- [ ] **Step 4: Verify the temporary tests pass**

Run the same pytest command. Expected: all tests pass.

### Task 2: Verify the Beetle package and remove temporary tests

**Files:**
- Remove: `src/runner/nodes/training/beetle/training/_temporary_test_signal_checkpoint.py`
- Verify: `src/runner/nodes/training/beetle/training/loop.py`
- Verify: `src/runner/nodes/training/beetle/training/loop_events.py`

**Interfaces:**
- Consumes: completed Task 1 implementation.
- Produces: compiled production code with no committed tests or temporary files.

- [ ] **Step 1: Compile Beetle through Nix**

Run `nix develop --command python -m compileall -q src/runner/nodes/training/beetle`.

Expected: exit status 0.

- [ ] **Step 2: Remove the temporary test with `apply_patch`**

Delete the temporary test file after successful verification.

- [ ] **Step 3: Verify final state**

Run `test ! -e src/runner/nodes/training/beetle/training/_temporary_test_signal_checkpoint.py`, then `git diff --check`, then inspect the two production diffs.

Expected: no temporary test remains, whitespace validation passes, and the diff
contains only boundary polling and cancellation checkpoint behavior.
