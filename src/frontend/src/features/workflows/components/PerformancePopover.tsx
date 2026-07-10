import { useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import { useWorkflowStore } from "../store";
import type { NodeRunSnapshot } from "../types";

type SortKey = "total" | "queue" | "resource" | "p95" | "throughput";

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: "total", label: "active total" },
  { key: "queue", label: "queue avg" },
  { key: "resource", label: "resource avg" },
  { key: "p95", label: "p95 batch" },
  { key: "throughput", label: "items/s" },
];

export function PerformancePopover({ onClose }: { onClose: () => void }) {
  const [sort, setSort] = useState<SortKey>("total");
  const [now, setNow] = useState(Date.now());
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const { activeRunId, snapshots } = useWorkflowStore();
  const snapshot = activeRunId ? snapshots[activeRunId] : undefined;
  const nodes = useMemo(() => {
    const rows = [...(snapshot?.nodes ?? [])];
    rows.sort((left, right) => sortValue(right, sort, now) - sortValue(left, sort, now));
    return rows;
  }, [now, snapshot, sort]);
  const hasRunningBatch = snapshot?.nodes.some((node) => node.performance.current_batch_started_at !== null) ?? false;
  useEffect(() => {
    if (!hasRunningBatch) return;
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [hasRunningBatch]);
  const virtualizer = useVirtualizer({
    count: nodes.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 70,
    overscan: 8,
  });

  return (
    <section className="absolute bottom-14 left-0 w-[min(960px,calc(100vw-2rem))] overflow-hidden rounded-lg border border-line bg-panel shadow-2xl">
      <header className="flex items-center justify-between border-b border-line px-3 py-2">
        <div>
          <strong className="text-[13px] text-txt">Node performance</strong>
          <span className="ml-2 font-mono text-[10px] text-txt-mute">{activeRunId ?? "no run selected"}</span>
        </div>
        <button type="button" className="cursor-pointer text-lg text-txt-mute hover:text-txt" onClick={onClose} aria-label="Close performance">&times;</button>
      </header>
      <div className="grid grid-cols-[minmax(150px,1.4fr)_repeat(5,minmax(76px,0.7fr))_74px] border-b border-line bg-panel-2 px-3 py-1.5 font-mono text-[9px] font-bold uppercase text-txt-mute">
        <span>node / recent batches</span>
        {COLUMNS.map((column) => (
          <button key={column.key} type="button" className={`cursor-pointer text-right uppercase ${sort === column.key ? "text-amber-700" : "hover:text-txt"}`} onClick={() => setSort(column.key)}>
            {column.label}
          </button>
        ))}
        <span className="text-right">max q</span>
      </div>
      <div ref={scrollRef} className="h-[min(540px,60vh)] overflow-auto">
        {nodes.length === 0 ? <div className="p-8 text-center text-sm text-txt-mute">Run a graph or select a completed run to see performance.</div> : null}
        <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
          {virtualizer.getVirtualItems().map((item) => {
            const node = nodes[item.index]!;
            return <PerformanceRow key={node.node_id} node={node} top={item.start} now={now} />;
          })}
        </div>
      </div>
      <footer className="flex flex-wrap gap-x-3 border-t border-line px-3 py-1.5 font-mono text-[9px] text-txt-mute">
        <Legend color="bg-slate-300" label="queue" /><Legend color="bg-violet-400" label="resource" />
        <Legend color="bg-blue-400" label="load" /><Legend color="bg-emerald-500" label="execute" />
        <Legend color="bg-cyan-400" label="route" />
      </footer>
    </section>
  );
}

function PerformanceRow({ node, top, now }: { node: NodeRunSnapshot; top: number; now: number }) {
  const metrics = node.performance;
  const batches = metrics.batches || 1;
  const waitBatches = metrics.batches + (metrics.current_batch_started_at ? 1 : 0) || 1;
  const currentElapsed = metrics.current_batch_started_at ? Math.max(0, now - Date.parse(metrics.current_batch_started_at)) : 0;
  const completedTotal = metrics.total_resource_wait_ms + metrics.total_load_ms + metrics.total_execute_ms + metrics.total_unload_ms + metrics.total_route_ms;
  return (
    <div className="absolute left-0 grid w-full grid-cols-[minmax(150px,1.4fr)_repeat(5,minmax(76px,0.7fr))_74px] items-center border-b border-line px-3 py-2 font-mono text-[10px]" style={{ height: 70, transform: `translateY(${top}px)` }}>
      <div className="min-w-0 pr-3">
        <strong className="block truncate text-[11px] text-txt">{node.node_id}</strong>
        <BatchTimeline node={node} />
      </div>
      <Value value={duration(completedTotal + currentElapsed)} warning={currentElapsed > metrics.p95_batch_ms && metrics.p95_batch_ms > 0} />
      <Value value={duration((metrics.total_queue_wait_ms + metrics.current_queue_wait_ms) / waitBatches)} />
      <Value value={duration(metrics.total_resource_wait_ms / batches)} />
      <Value value={duration(metrics.p95_batch_ms)} />
      <Value value={rate(metrics.items_per_second)} />
      <Value value={String(metrics.max_queue_size)} warning={node.queue_size > 0 && node.running_batches > 0} />
    </div>
  );
}

function BatchTimeline({ node }: { node: NodeRunSnapshot }) {
  const recent = node.performance.recent_batches.slice(-12);
  return (
    <div className="mt-1 flex h-3 gap-px" title="Latest batches: queue, resource, load, execute, route">
      {recent.map((batch) => {
        const total = batch.queue_wait_ms + batch.resource_wait_ms + batch.load_ms + batch.execute_ms + batch.unload_ms + batch.route_ms || 1;
        return (
          <div key={batch.batch_index} className="flex min-w-[5px] flex-1 overflow-hidden rounded-sm" title={`batch ${batch.batch_index} · ${duration(batch.total_ms)}`}>
            <Phase color="bg-slate-300" value={batch.queue_wait_ms / total} />
            <Phase color="bg-violet-400" value={batch.resource_wait_ms / total} />
            <Phase color="bg-blue-400" value={(batch.load_ms + batch.unload_ms) / total} />
            <Phase color="bg-emerald-500" value={batch.execute_ms / total} />
            <Phase color="bg-cyan-400" value={batch.route_ms / total} />
          </div>
        );
      })}
    </div>
  );
}

function Phase({ color, value }: { color: string; value: number }) {
  return <span className={color} style={{ width: `${value * 100}%` }} />;
}

function Value({ value, warning = false }: { value: string; warning?: boolean }) {
  return <span className={`text-right ${warning ? "font-bold text-amber-700" : "text-txt-dim"}`}>{value}</span>;
}

function Legend({ color, label }: { color: string; label: string }) {
  return <span className="flex items-center gap-1"><i className={`h-2 w-2 rounded-sm ${color}`} />{label}</span>;
}

function sortValue(node: NodeRunSnapshot, key: SortKey, now: number): number {
  const metrics = node.performance;
  const batches = metrics.batches || 1;
  if (key === "queue") {
    const waitBatches = metrics.batches + (metrics.current_batch_started_at ? 1 : 0) || 1;
    return (metrics.total_queue_wait_ms + metrics.current_queue_wait_ms) / waitBatches;
  }
  if (key === "resource") return metrics.total_resource_wait_ms / batches;
  if (key === "p95") return metrics.p95_batch_ms;
  if (key === "throughput") return metrics.items_per_second;
  const currentElapsed = metrics.current_batch_started_at ? Math.max(0, now - Date.parse(metrics.current_batch_started_at)) : 0;
  return metrics.total_resource_wait_ms + metrics.total_load_ms + metrics.total_execute_ms + metrics.total_unload_ms + metrics.total_route_ms + currentElapsed;
}

function duration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds.toFixed(0)}ms`;
  if (milliseconds < 60_000) return `${(milliseconds / 1000).toFixed(1)}s`;
  return `${(milliseconds / 60_000).toFixed(1)}m`;
}

function rate(value: number): string {
  return value < 10 ? value.toFixed(1) : value.toFixed(0);
}
