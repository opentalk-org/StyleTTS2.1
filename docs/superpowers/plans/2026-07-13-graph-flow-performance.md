# Graph Flow Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous execute-time rates with real end-to-end graph throughput, exact flow/backpressure telemetry, and a neutral node/edge heatmap that makes bottlenecks inspectable without guessing for the user.

**Architecture:** The scheduler will assign one stable root lineage to each source item and reference-count its queued, running, routed, and join-buffer work until every terminal branch is complete. Runtime events feed a bounded shared performance accumulator that publishes graph, node, and edge snapshots; the editor renders those values directly as a graph strip, compact node summaries, neutral heatmap, and virtualized detail table.

**Tech Stack:** Python 3.12, Pydantic 2, asyncio, FastAPI snapshot/event pipeline, React 19, TypeScript, Zustand, TanStack Virtual, Tailwind CSS v4.

## Global Constraints

- `src/runflow` remains domain-agnostic; use item, lineage, edge, queue, batch, task, and resource terminology.
- Completed graph throughput means source lineages per wall-clock second, counted once only after every terminal fan-out branch has finished.
- Publish both a rolling 30-second rate and a whole-run average; retain at most 60 one-second history buckets.
- Use a neutral heatmap and factual measurements only; do not calculate or display an automatic bottleneck label.
- Preserve entity/artifact identity separately from the stable root lineage after the source boundary.
- Do not add snapshot compatibility fallbacks or migration logic for the previous performance schema.
- Use bounded constant-size state for rates and latency distributions.
- Use temporary tests outside the repository and remove them after verification; test runtime behavior through a real submitted graph.
- Run every Python, frontend, and service command through `nix develop --command ...`.
- Work in the current checkout without a branch, worktree, or commit; preserve unrelated dirty changes.
- Do not dispatch subagents in this session; execute inline with `superpowers:executing-plans` because delegation was not requested.

## File Map

- Create `src/runflow/runtime/lineage_tracker.py`: root-lineage reference counting and completion/abandonment transitions.
- Modify `src/runflow/runtime/routing.py`: return typed join reservation changes with ready tasks.
- Modify `src/runflow/runtime/output_router.py`: preserve root lineage, emit packet/edge events, and connect routing to lineage accounting.
- Modify `src/runflow/runtime/scheduler.py`: own the tracker, time queue blocking, and close task references reliably.
- Modify `src/runflow/runtime/scheduler_events.py`: emit typed lineage and exact edge telemetry.
- Create `src/shared/performance_state.py`: bounded graph/node/edge aggregation and snapshot conversion.
- Modify `src/shared/event_store.py`: delegate performance events while retaining node lifecycle state.
- Modify `src/shared/schemas.py`: replace ambiguous metrics with explicit graph, node, edge, rate-history, and latency fields.
- Modify `src/frontend/src/features/workflows/types.ts`: mirror the new snapshot contract.
- Create `src/frontend/src/features/workflows/performance.ts`: pure formatting, edge lookup, neutral intensity, and sorting helpers.
- Create `src/frontend/src/features/workflows/components/NodePerformanceSummary.tsx`: compact node flow/backlog display.
- Create `src/frontend/src/features/workflows/components/RunPerformanceStrip.tsx`: graph-wide live and whole-run flow summary.
- Modify `src/frontend/src/features/workflows/components/WorkflowNodeCard.tsx`: use the compact summary and neutral queue-fill background.
- Modify `src/frontend/src/features/workflows/components/WorkflowEdges.tsx`: render flow width/speed plus backpressure/resource-wait markers.
- Modify `src/frontend/src/features/workflows/components/PerformancePopover.tsx`: virtualized factual node table with graph-order and numeric sorts.
- Modify `src/frontend/src/features/workflows/components/WorkflowRunPanel.tsx`: mount the graph summary strip.
- Modify `src/frontend/src/index.css`: add the single edge-flow dash animation.
- Update `docs/superpowers/specs/2026-07-13-graph-flow-performance-design.md` only if implementation reveals a semantic correction.

---

### Task 1: Stable source lineage accounting

**Files:**
- Create: `src/runflow/runtime/lineage_tracker.py`
- Test: `/tmp/test_lineage_tracker.py`

**Interfaces:**
- Produces `LineageTracker(events: SchedulerEventEmitter)`, `tracks(lineage_id)`, `open_source(lineage_id, source_node_id)`, `close_source(lineage_id)`, `add_task(lineage_id)`, `finish_task(lineage_id)`, `open_join(lineage_id)`, `close_join(lineage_id)`, and `abandon_all(reason)`.
- A lineage completes exactly when `source_open == 0`, `tasks == 0`, and `joins == 0`; methods raise on an unknown lineage or negative count.

- [ ] **Step 1: Write the failing tracker test** with an async fake emitter and three cases: a two-branch fan-out emits one completion only after both `finish_task` calls; an open join reservation prevents early completion; `abandon_all` emits one abandonment per active lineage and clears the tracker.

```python
class Events:
    def __init__(self): self.transitions = []
    async def lineage_started(self, lineage_id, source_node_id): self.transitions.append(("started", lineage_id))
    async def lineage_completed(self, lineage_id, elapsed_ms): self.transitions.append(("completed", lineage_id))
    async def lineage_abandoned(self, lineage_id, reason): self.transitions.append(("abandoned", lineage_id))

async def test_fanout_completes_once():
    events = Events(); tracker = LineageTracker(events)
    await tracker.open_source("source:a", "source"); await tracker.add_task("source:a"); await tracker.add_task("source:a")
    await tracker.close_source("source:a"); await tracker.finish_task("source:a")
    assert events.transitions == [("started", "source:a")]
    await tracker.finish_task("source:a")
    assert events.transitions == [("started", "source:a"), ("completed", "source:a")]
```

- [ ] **Step 2: Verify red.** Run `nix develop --command python -m unittest /tmp/test_lineage_tracker.py -v`; expect import failure for `runflow.runtime.lineage_tracker`.
- [ ] **Step 3: Implement the tracker** with a module-level `LineageState` dataclass containing `source_node_id`, `started_at: float`, `source_open: int`, `tasks: int`, and `joins: int`. Store active states in `dict[str, LineageState]`; use `perf_counter()` for elapsed time and one private `_complete_if_idle()` transition.
- [ ] **Step 4: Verify green.** Re-run the command; expect all three tests to pass.

### Task 2: Route every branch without losing lineage

**Files:**
- Modify: `src/runflow/runtime/routing.py`
- Modify: `src/runflow/runtime/output_router.py`
- Modify: `src/runflow/runtime/scheduler.py`
- Test: `/tmp/test_runtime_lineage.py`

**Interfaces:**
- `JoinRoutingResult(tasks: list[Task], opened_lineages: tuple[str, ...], closed_lineages: tuple[str, ...])` replaces the raw task list from `add_to_join_buffer`.
- `OutputRouter(..., enqueue: Callable[[str, Task], Awaitable[float]], lineages: LineageTracker)` consumes Task enqueue block time in milliseconds.
- Source outputs use `f"{node.id}:{lineage_from_value(item, inherited=f'{batch_index}:{output_index}')}"`; non-source outputs always use `task.lineage_id`. The source-node namespace prevents equal entity ids from colliding across sources; the item/entity id remains available through packet value/metadata and never replaces a downstream root lineage.

- [ ] **Step 1: Write failing runtime tests** using registered minimal source, pass-through, join, and sink nodes in a real `Graph`/`Scheduler`: assert one source item fanned to two sinks emits one `lineage_completed`; a join holds completion between its first and final input; downstream values with their own `id` do not change the root lineage.
- [ ] **Step 2: Verify red.** Run `nix develop --command python -m unittest /tmp/test_runtime_lineage.py -v`; expect missing `JoinRoutingResult`/completion events.
- [ ] **Step 3: Implement typed join transitions.** For the first item packet retained in `state.item_groups[lineage_id]`, report that lineage in `opened_lineages`; when the group is fully drained into ready tasks, report it in `closed_lineages`. Broadcast packets are context and do not create a counted source lineage; the item lineage remains the unit of work.
- [ ] **Step 4: Integrate routing accounting.** Open a source lineage before routing all packets for that source output and close it afterward in `finally`; call `add_task` before enqueue and undo with `finish_task` if enqueue raises; open join reservations before accepting a waiting packet, add ready task references, then close drained reservations after enqueue ownership exists.
- [ ] **Step 5: Integrate scheduler completion.** Change `_enqueue()` to return measured `queue.put()` block milliseconds, emit `task_enqueued` after insertion, pass the completed `Task` to `_mark_task_done(task)`, and call `finish_task(task.lineage_id)` in the worker's existing per-task completion `finally` only when `tracks(task.lineage_id)` is true. Source-control tasks such as `input:<node>:<window>` are scheduler work, not source items, and must not enter graph-throughput counts. On scheduler failure/cancellation call `abandon_all()` once before propagating.
- [ ] **Step 6: Verify green and compile.** Run `nix develop --command python -m unittest /tmp/test_runtime_lineage.py -v` and `nix develop --command python -m compileall -q src/runflow`; expect all tests and compilation to pass.

### Task 3: Emit exact graph-flow events

**Files:**
- Modify: `src/runflow/runtime/scheduler_events.py`
- Modify: `src/runflow/runtime/output_router.py`
- Test: `/tmp/test_scheduler_flow_events.py`

**Interfaces:**
- Emits `lineage_started {source_node_id}`, `lineage_completed {elapsed_ms}`, and `lineage_abandoned {reason}`.
- Emits one `packet_created` per packet and one `packet_delivered` per traversed edge with `detail={..., "enqueue_blocked_ms": float}`.
- `join_waiting` includes `waiting_lineages: int`; `batch_started` includes queue capacity; `batch_completed` retains all phase timings.

- [ ] **Step 1: Write a failing emitter/router test** that captures `ExecutionContext.emit_event`, routes one packet over two edges, and asserts one created event, two delivered events, stable lineage ids, and non-negative `enqueue_blocked_ms` values.
- [ ] **Step 2: Verify red.** Run `nix develop --command python -m unittest /tmp/test_scheduler_flow_events.py -v`; expect absent created/delivered calls or missing detail.
- [ ] **Step 3: Add explicit emitter methods and calls.** Emit `packet_created` immediately after packet construction. Emit `packet_delivered` after direct enqueue; for joins emit delivery when the packet enters the join and attribute any queue block to the edge whose packet releases the ready task. This makes edge blocking causal and avoids dividing one wait across unrelated edges.
- [ ] **Step 4: Verify green.** Re-run the test; expect all event assertions to pass.

### Task 4: Bounded graph, node, and edge aggregation

**Files:**
- Create: `src/shared/performance_state.py`
- Modify: `src/shared/schemas.py`
- Modify: `src/shared/event_store.py`
- Test: `/tmp/test_performance_state.py`

**Interfaces:**
- `RatePoint(timestamp: datetime, count: int, rate: float)`; `GraphPerformanceSnapshot(started_items, completed_items, inflight_items, abandoned_items, rolling_throughput, average_throughput, latency_p50_ms, latency_p95_ms, history)`.
- `NodePerformanceSnapshot(arrival_rate, departure_rate, arrived_items, departed_items, queue_capacity, queue_fill_ratio, queue_growth_rate, busy_ratio, resource_wait_ratio, downstream_blocked_ms, batch_p50_ms, batch_p95_ms, service_capacity, current_batch_started_at, recent_batches)`.
- `EdgePerformanceSnapshot(source_node, source_port, target_node, target_port, delivered_items, rolling_rate, enqueue_blocked_ms, join_waiting_items)`.
- `RunSnapshot` adds `performance: GraphPerformanceSnapshot` and `edges: list[EdgePerformanceSnapshot]`.
- `PerformanceState.record(event)` and `PerformanceState.snapshot(now)` are the only integration points used by `RunEventStore`.

- [ ] **Step 1: Write failing aggregation tests** with fixed UTC timestamps. Assert `started == completed + inflight + abandoned`; exactly 30 seconds contribute to rolling rate; the 60-bucket history evicts older seconds; node arrival/departure rates use wall time; queue growth is signed; busy/resource ratios use run wall time; edge delivery and enqueue blocking preserve the four endpoint fields; p50/p95 are deterministic.
- [ ] **Step 2: Verify red.** Run `nix develop --command python -m unittest /tmp/test_performance_state.py -v`; expect missing `PerformanceState` and schema fields.
- [ ] **Step 3: Define strict Pydantic snapshot models** and remove `input_items_per_second`/`output_items_per_second`. Keep `BatchPerformanceSnapshot` only for the 30 recent phase rows. Do not add aliases, optional old fields, or parsing fallbacks.
- [ ] **Step 4: Implement bounded accumulators.** Use a fixed `deque(maxlen=60)` of UTC second buckets for graph/node/edge counts and a deterministic 512-entry reservoir for whole-run lineage/batch latency. Calculate snapshots from event timestamps plus `now`; clamp only mathematical ratios to `[0, 1]`, while queue growth stays signed.
- [ ] **Step 5: Delegate from the event store.** Move performance-only batch state out of `event_store.py`, call `self.performance.record(event)` from `RunEventStore.record`, and include graph/edge snapshots while retaining lifecycle/error/node counters. Ensure each touched Python file remains below 300 lines.
- [ ] **Step 6: Verify green and schema parsing.** Re-run the test and `nix develop --command python -m compileall -q src/shared src/backend src/runner`; expect all tests and compilation to pass.

### Task 5: Frontend metric contract and neutral visual helpers

**Files:**
- Modify: `src/frontend/src/features/workflows/types.ts`
- Create: `src/frontend/src/features/workflows/performance.ts`
- Test: frontend type-check/build in Task 7

**Interfaces:**
- Types mirror every Task 4 field without optional legacy members.
- Exports `edgeKey(edge)`, `nodePerformance(snapshot, nodeId)`, `edgePerformance(snapshot, edge)`, `formatRate`, `formatDuration`, `neutralQueueTone(fillRatio)`, `flowStrokeWidth(rate, maximumRate)`, and `performanceSortValue(node, key)`.

- [ ] **Step 1: Replace the TypeScript snapshot contract** exactly: graph metrics live at `snapshot.performance`, edge metrics at `snapshot.edges`, and node performance uses the explicit wall-rate/capacity names from Task 4.
- [ ] **Step 2: Implement pure helpers.** Queue tones interpolate neutral slate opacity from `0.00` at empty to `0.18` at full; edge width maps the run-relative rate to `1.5..6` pixels using square-root scaling; format zero as `0/s`, rates below ten with one decimal, and durations below one second as integer milliseconds.
- [ ] **Step 3: Check file limits.** Run `wc -l src/frontend/src/features/workflows/types.ts src/frontend/src/features/workflows/performance.ts`; expect both below 300.

### Task 6: Graph strip, compact node summary, and edge heatmap

**Files:**
- Create: `src/frontend/src/features/workflows/components/NodePerformanceSummary.tsx`
- Create: `src/frontend/src/features/workflows/components/RunPerformanceStrip.tsx`
- Modify: `src/frontend/src/features/workflows/components/WorkflowNodeCard.tsx`
- Modify: `src/frontend/src/features/workflows/components/WorkflowEdges.tsx`
- Modify: `src/frontend/src/features/workflows/components/WorkflowRunPanel.tsx`
- Modify: `src/frontend/src/index.css`

**Interfaces:**
- `NodePerformanceSummary({ performance })` renders `in → out`, `queue used/capacity signed-growth`, and `busy · resource wait`.
- `RunPerformanceStrip({ snapshot })` renders rolling/average throughput, completed/started, inflight, p50, and p95.

- [ ] **Step 1: Add compact components** with exact labels: `27.5/s in → 18.2/s out`, `queue 410/512 ↑9.3/s`, `busy 96% · resource wait 3%`; render em dashes only before a run has any observations.
- [ ] **Step 2: Replace the node's old four-column service-rate block** with `NodePerformanceSummary`; apply `neutralQueueTone(queue_fill_ratio)` as a subtle background overlay and retain failed/running/stopped lifecycle borders unchanged.
- [ ] **Step 3: Render factual edge heat.** Set width from rolling delivery rate relative to the busiest visible edge; set dash animation duration inversely from rate; show a small neutral arrow for positive target queue growth, a violet tick when target resource wait is nonzero, and a square marker when `enqueue_blocked_ms > 0`. Do not add red/amber bottleneck judgments.
- [ ] **Step 4: Mount `RunPerformanceStrip`** inside the existing run panel before controls and allow the panel to wrap on narrow viewports.
- [ ] **Step 5: Add one CSS keyframe** named `workflow-edge-flow` that shifts `stroke-dashoffset`; respect `prefers-reduced-motion` by disabling edge animation.
- [ ] **Step 6: Check component count and file limits.** Run `find src/frontend/src/features/workflows/components -maxdepth 1 -type f | wc -l` and `wc -l` for all touched components; expect at most 16 component files and every file below 300 lines.

### Task 7: Virtualized performance inspector

**Files:**
- Modify: `src/frontend/src/features/workflows/components/PerformancePopover.tsx`

**Interfaces:**
- Default order follows `graph.nodes`; sort keys are `arrival`, `departure`, `queueFill`, `queueGrowth`, `busy`, `resourceWait`, `blocked`, and `p95`.
- Clicking a row calls `selectNode(node.node_id)` and preserves virtualization.

- [ ] **Step 1: Replace ambiguous columns** with `in/s`, `out/s`, `queue`, `Δ queue/s`, `busy`, `resource wait`, `blocked`, and `p95 batch`; add the graph strip above the header row. Keep `@tanstack/react-virtual`, 70px row estimates, and eight-row overscan.
- [ ] **Step 2: Add graph-order and numeric sorting.** The first header option restores graph order; numeric headers toggle descending/ascending. Empty runs show the existing explanatory state.
- [ ] **Step 3: Add row focus.** Read `selectNode` from the store and focus the corresponding node on row click; use factual violet only for resource wait and neutral slate for all flow/queue values.
- [ ] **Step 4: Build frontend.** Run `nix develop --command bash -lc 'cd src/frontend && npm run build'`; expect TypeScript and Vite to finish successfully with no missing old metric fields.

### Task 8: Full-stack verification through a real graph

**Files:**
- Temporary: `/tmp/graph-flow-performance.json`
- No committed test or fixture files.

- [ ] **Step 1: Run focused verification.** Execute all four temporary Python test files together through `nix develop --command python -m unittest ... -v`; expect every test to pass. Run `nix develop --command python -m compileall -q src/runflow src/shared src/backend src/runner` and the frontend build; expect zero errors.
- [ ] **Step 2: Restart the shared stack once.** Run `nix develop --command runflow-dev-stop`, then as the normal workspace user run `nix develop --command runflow-dev-session`, detach from Zellij, and check `nix develop --command runflow-dev-status`; expect one backend, one frontend, one NATS service, and the configured runner online.
- [ ] **Step 3: Submit a fan-out smoke graph.** Create a temporary inline request with one registered `TestingRunInput` source feeding two registered sink-capable testing nodes, identical source lineage on both edges, default `RunContextRequest`, and a unique `run_id`; submit it to `POST /graphs/runs` exactly as the UI does.
- [ ] **Step 4: Inspect through supported interfaces.** Run `nix develop --command python -m cli runs`, `nix develop --command python -m cli logs <run_id>`, and query the run snapshot endpoint. Expect terminal success, `started_items == 1`, `completed_items == 1`, `inflight_items == 0`, `abandoned_items == 0`, two populated edge rows, non-negative rolling/average throughput, and no duplicate lineage completion.
- [ ] **Step 5: Verify the UI manually.** Select the smoke run and confirm the graph strip reports completed source items, both outgoing edges animate with neutral flow widths, nodes show wall rates/queue/busy/resource wait, the popover sorts and focuses nodes, and no automatic bottleneck label appears.
- [ ] **Step 6: Remove temporary artifacts.** Delete `/tmp/test_lineage_tracker.py`, `/tmp/test_runtime_lineage.py`, `/tmp/test_scheduler_flow_events.py`, `/tmp/test_performance_state.py`, and `/tmp/graph-flow-performance.json`; remove the smoke run only if the existing CRUD/CLI offers an explicit safe delete operation.
- [ ] **Step 7: Review scope.** Run `git status --short` and `git diff --check`; confirm only the planned files changed in this feature, no generated frontend artifacts were added, and unrelated user changes remain untouched.

## Completion Criteria

- A source item is counted exactly once only after all fan-out and join work has terminated.
- Restart/failure/cancellation cannot leave an active lineage counted as completed; it becomes abandoned.
- The 30-second rate, whole-run rate, p50/p95, node rates, queue growth/fill, resource wait, and edge blocking are based on wall-clock event data with bounded storage.
- The UI distinguishes real graph throughput from node service capacity in its labels and descriptions.
- The graph remains readable without color judgment: intensity, width, motion, arrows, and small markers encode measurements; no node is declared the bottleneck.
- Python verification, frontend production build, shared-stack health, and the real fan-out graph all pass.
