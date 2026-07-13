# Node Build Failure Highlighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist graph-build validation failures against their exact workflow node so the existing frontend failed-node styling activates.

**Architecture:** Convert `GraphNodeBuildError` into the existing `RunEvent(kind="node_failed")` at the `RunExecution` boundary, then preserve the existing job terminal update. The event store remains the sole owner of node snapshot projection.

**Tech Stack:** Python 3.11+, asyncio, dataclasses, unittest, Pydantic, Nix development shell

## Global Constraints

- Do not support legacy graph formats or legacy runner transport behavior.
- Keep `src/runflow` domain-agnostic.
- Run Python and pytest only through `nix develop --command`.
- Remove the throwaway regression test before finishing.

---

### Task 1: Record graph-build failures against the node

**Files:**
- Modify: `src/runner/run_execution.py:46-49`
- Test temporarily: `/tmp/test_run_execution_node_failure.py`

**Interfaces:**
- Consumes: `GraphNodeBuildError.node_id`, `GraphNodeBuildError.node_type`, and `RunnerStateBuffer.event_sink(event: RunEvent)`
- Produces: one `RunEvent` with `kind="node_failed"`, the failing `node_id`, the validation message, and graph-build diagnostic detail

- [ ] **Step 1: Write the failing test**

Create a temporary async test that patches `build_inline_graph` to raise `GraphNodeBuildError("writeback", "AddAudioToDataset", validation_error)`, executes `RunExecution`, and asserts that one captured event has `kind == "node_failed"` and `node_id == "writeback"`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `nix develop --command python -m unittest discover -s /tmp -p 'test_run_execution_node_failure.py' -v`

Expected: FAIL because `RunExecution` currently marks only the job terminal and records no event.

- [ ] **Step 3: Write the minimal implementation**

In the `GraphNodeBuildError` handler, construct a `RunEvent` with the node ID, node type, `stage="graph_build"`, and traceback, pass it to `self.state.event_sink`, then call `mark_terminal` with the same error message.

- [ ] **Step 4: Run focused and broader verification**

Run: `nix develop --command python -m unittest discover -s /tmp -p 'test_run_execution_node_failure.py' -v`

Expected: PASS.

Run the repository's available Python checks covering runner/shared code through `nix develop`; expected exit code is 0.

- [ ] **Step 5: Remove the temporary test and inspect the diff**

Delete `/tmp/test_run_execution_node_failure.py`, confirm only the scoped production and planning files changed, and do not modify unrelated user work.
