# Scalable Buffering Design

## Goal

Make batching observable and scalable for many small payloads and a few large payloads without adding domain knowledge to `runflow`.

## Runtime boundary

`runflow` owns task buffering, task-cost admission, batching, backpressure, and scheduler events. The node queue is the only waiting-task buffer. A batch collector may wait for readiness by observing queue depth, upstream activity, deadlines, and budget pressure, but it must not remove tasks until it has decided to start work. Consequently, `queue_size` includes every task waiting for execution.

Task cost remains generic and derives from payload byte hints or resident values. Batch readiness ends when the preferred count is available, the deadline expires, upstream cannot produce more tasks, or a producer is blocked by the target buffer's count or byte capacity. No decision depends on node type, port datatype, or graph-node name.

Scheduler task counts and node work-item counts are distinct. `batch_completed` reports `completed_items`: source nodes report emitted output items, while transforms report consumed input items. The UI's `done` counter uses this explicit value.

## Audio boundary

Audio storage decides how stored WAV ranges are read. Segment references are paged by estimated output-clip bytes and count, not by the size of complete source recordings. The reader groups small recordings for the existing bulk path and uses object-store byte ranges for recordings larger than the resident read budget. Several ranges from the same recording share one parsed WAV layout.

This logic stays under `src/shared/db/audio` and `src/runner`; `runflow` receives ordinary typed task payloads and their byte costs.

## Verification

Temporary regressions cover a slow producer building a preferred batch while queue depth remains visible, byte-pressure flushing, source completed-item counters, many small WAV sources, and large WAV range reads. A real `SpeakerSegmentSource -> ECAPASpeakerEmbed` graph must show non-singleton source pages after the initial pages and 128-item embedding batches. Temporary tests are removed before handoff.
