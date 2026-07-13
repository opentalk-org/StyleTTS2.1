# Graph Flow Performance Design

## Purpose

Replace node-local cumulative timing displays with metrics that answer three operational questions:

1. How much useful work is the complete graph finishing now and over the whole run?
2. Where is work accumulating or flow being restricted?
3. Is a node busy, starved, waiting for a resource, or blocked by downstream capacity?

The UI presents neutral measurements through graph topology. It does not calculate or display an automatic bottleneck label or opaque composite score.

## Metric correctness

### End-to-end work unit

The primary graph throughput unit is a completed source lineage per second. A source lineage represents one item emitted by an input node, namespaced by the source node. It is counted once when all work derived from that item has cleared terminal branches and join buffers.

Entity IDs and artifact IDs are not lineage IDs. The runtime carries a stable root lineage separately so transforms may create new entities without breaking end-to-end correlation.

### Completion tracking

The scheduler owns lineage completion because it owns tasks, routing, joins, and graph idleness. For each active source lineage it tracks:

- source entry time;
- outstanding queued or executing tasks;
- routed packets held in join buffers;
- source routing in progress;
- terminal branch completion.

A lineage completes only when source routing has closed and its outstanding task and buffered-packet counts are zero. Fan-out increments outstanding work for each branch. Join packets remain outstanding until consumed into a task or explicitly discarded by graph semantics. A source-only graph completes an item after its terminal source output has been handled.

Completed lineage state is removed immediately after contributing to aggregates. Active state is bounded by current in-flight work rather than total run size. The invariant is `started = completed + in_flight + abandoned`.

### Time horizons

Every flow metric has two views:

- **Live:** rolling 30-second values built from one-second buckets.
- **Run:** cumulative values from run start through the snapshot time.

The snapshot retains at most 60 one-second graph-history buckets for the headline sparkline. It does not retain per-item or unbounded batch history.

## Graph metrics

The run snapshot exposes:

- `lineages_started`: total source items admitted;
- `lineages_completed`: source items completed through the full graph;
- `lineages_in_flight`: source items with outstanding work, including join waits;
- `lineages_abandoned`: incomplete source items cleared by failure or stop;
- `rolling_completed_per_second`: completions in the rolling window divided by observed window duration;
- `average_completed_per_second`: completions divided by run wall time;
- `end_to_end_latency_p50_ms` and `end_to_end_latency_p95_ms` from bounded sampling;
- `throughput_history`: up to 60 timestamped completion-rate buckets.

Warm-up windows divide by their actual observed duration rather than always dividing by 30 seconds. A stopped or completed run retains its final values without fabricating activity after its terminal timestamp.

## Node metrics

Node flow metrics distinguish wall-clock flow from service capacity:

- `rolling_arrival_per_second`: input tasks admitted per wall-clock second;
- `rolling_departure_per_second`: output items routed per wall-clock second, or consumed items for terminal nodes;
- `average_arrival_per_second` and `average_departure_per_second`: whole-run wall-clock rates;
- `busy_ratio`: worker busy time divided by available worker time;
- `queue_size`, `queue_capacity`, and `queue_fill_ratio`;
- `queue_growth_per_second`: rolling queue-size change divided by observed duration;
- `resource_wait_ratio`: worker time awaiting declared resources divided by active worker time;
- `backpressure_ratio`: worker time blocked enqueueing into downstream bounded queues divided by active worker time;
- `batch_latency_p50_ms` and `batch_latency_p95_ms`;
- `service_capacity_per_second`: processed output items divided by execute time, explicitly named as capacity rather than throughput.

Busy, wait, and backpressure ratios use worker-time denominators so future node concurrency remains meaningful. Queue growth uses observed queue-depth samples, not batch averages.

## Edge metrics

Each graph edge exposes:

- rolling and whole-run delivered items per second;
- total delivered items;
- rolling enqueue-blocked time;
- current join-wait item count when the target port participates in a join.

The router attributes bounded-queue blocking to the exact source/target edge. General route processing time is not labeled as backpressure.

## Runtime events and aggregation

The runtime emits structured flow events at ownership boundaries:

- source lineage admitted and completed;
- task enqueued and completed;
- edge item delivered with enqueue-blocked duration;
- join item buffered and consumed;
- batch started and completed with worker timing.

`RunEventStore` reduces these events into constant-size graph, node, and edge snapshots. PostgreSQL persists only the reduced snapshot through the existing runner state flush. No metrics database, frontend recomputation from logs, or domain-specific scheduler behavior is introduced.

## User interface

### Run summary strip

The workflow run panel shows:

- rolling end-to-end items/s as the primary value;
- whole-run average items/s;
- started, completed, and in-flight lineages;
- end-to-end latency p50/p95;
- a 60-second throughput sparkline.

Labels always include the horizon: `30s`, `run avg`, or `total`.

### Graph heatmap

The graph remains the primary diagnostic surface. Visual channels are independent and documented in a visible legend:

- node background temperature represents queue fill ratio;
- a node utilization bar represents busy ratio;
- edge thickness represents rolling delivered items/s;
- edge animation speed represents current delivery rate;
- a queue arrow and signed value represent growing, stable, or draining work;
- violet represents resource-wait ratio;
- a downstream-block marker represents backpressure ratio.

The default node summary is compact:

```text
27.5/s in → 18.2/s out
queue 410/512 ↑9.3/s
busy 96% · resource wait 3%
```

Input and terminal nodes omit inapplicable sides rather than showing misleading zeros. Colors never replace numeric values or direction indicators.

### Detail table

The performance popover becomes a virtualized, sortable table containing exact live and run metrics. Default columns are arrival/s, departure/s, busy percentage, queue size/capacity, queue growth/s, resource-wait percentage, backpressure percentage, batch p50/p95, service capacity, and completed count.

The table defaults to graph order. Users may sort by any numeric column, but no row is automatically named a bottleneck. Selecting a row focuses the corresponding graph node.

The current cumulative `active total`, ambiguous `input/s` and `output/s`, average queue wait, and phase-strip visualization are removed from the primary surface. Raw phase totals may remain in an inspector-only diagnostics section if they retain precise labels.

## Scale and storage

- One-second bucket rings are fixed-size.
- Latency quantiles use bounded samples or a bounded streaming estimator.
- Completed lineage records are discarded immediately.
- Active lineage memory scales with in-flight work.
- Frontend tables remain virtualized.
- Snapshot size scales with graph nodes, edges, and fixed history length, never with dataset item count or run duration.

## Failure and terminal behavior

Failed or stopped lineages do not count as completed useful work. Their active tracking moves to `lineages_abandoned` when the run becomes terminal. Final snapshots retain completed throughput and latency aggregates, report zero in-flight work, and set live flow to zero after the terminal timestamp.

Metrics must never prevent cancellation, routing, or terminal state persistence. Metric aggregation errors fail explicitly during development; the runtime does not silently substitute fabricated rates.

## Verification

Temporary tests and smoke graphs cover:

- linear graph completion;
- fan-out completion counted once after all branches;
- join-buffer work preventing premature completion;
- rolling-window expiration and warm-up denominators;
- wall-clock throughput differing from service capacity;
- exact edge backpressure attribution;
- bounded history and completed-lineage memory;
- terminal snapshots freezing live rates;
- neutral heatmap formatting and virtualized detail-table sorting.

Node/runtime tests run through real graphs as required by repository policy. Throwaway tests are removed before completion.
