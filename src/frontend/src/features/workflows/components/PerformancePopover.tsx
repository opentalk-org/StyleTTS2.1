import { useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import { formatDuration, formatRate, performanceSortValue, type PerformanceSortKey } from "../performance";
import { useWorkflowStore } from "../store";
import type { NodeRunSnapshot } from "../types";
import { RunPerformanceStrip } from "./RunPerformanceStrip";

type SortState = { key: PerformanceSortKey; ascending: boolean } | null;

const COLUMNS: { key: PerformanceSortKey; label: string }[] = [
  { key: "arrival", label: "in/s" },
  { key: "departure", label: "out/s" },
  { key: "queueFill", label: "queue" },
  { key: "queueGrowth", label: "Δ queue/s" },
  { key: "busy", label: "busy" },
  { key: "resourceWait", label: "resource wait" },
  { key: "blocked", label: "blocked" },
  { key: "p95", label: "p95 batch" },
];

export function PerformancePopover({ onClose }: { onClose: () => void }) {
  const [sort, setSort] = useState<SortState>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const { activeRunId, snapshots, graph, selectNode } = useWorkflowStore();
  const snapshot = activeRunId ? snapshots[activeRunId] : undefined;
  const nodes = useMemo(() => orderedNodes(snapshot?.nodes ?? [], graph.nodes.map((node) => node.id), sort), [graph.nodes, snapshot, sort]);
  const virtualizer = useVirtualizer({
    count: nodes.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 70,
    overscan: 8,
  });
  const changeSort = (key: PerformanceSortKey) => setSort((current) => (
    current?.key === key ? { key, ascending: !current.ascending } : { key, ascending: false }
  ));

  return (
    <section className="absolute bottom-14 left-0 w-[min(1180px,calc(100vw-2rem))] overflow-hidden rounded-lg border border-line bg-panel shadow-2xl">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-3 py-2">
        <div>
          <strong className="text-[13px] text-txt">Graph flow performance</strong>
          <span className="ml-2 font-mono text-[10px] text-txt-mute">{activeRunId ?? "no run selected"}</span>
        </div>
        <RunPerformanceStrip snapshot={snapshot} compact={false} />
        <button type="button" className="cursor-pointer text-lg text-txt-mute hover:text-txt" onClick={onClose} aria-label="Close performance">&times;</button>
      </header>
      <div className="overflow-x-auto">
        <div className="min-w-[1080px]">
          <div className="grid grid-cols-[minmax(150px,1.4fr)_repeat(8,minmax(82px,0.72fr))_88px] border-b border-line bg-panel-2 px-3 py-1.5 font-mono text-[9px] font-bold uppercase text-txt-mute">
            <button type="button" className="cursor-pointer text-left uppercase hover:text-txt" onClick={() => setSort(null)}>node / batches</button>
            {COLUMNS.map((column) => (
              <button key={column.key} type="button" className={`cursor-pointer text-right uppercase ${sort?.key === column.key ? "text-slate-800" : "hover:text-txt"}`} onClick={() => changeSort(column.key)}>
                {column.label}{sort?.key === column.key ? (sort.ascending ? " ↑" : " ↓") : ""}
              </button>
            ))}
            <span className="text-right">service cap</span>
          </div>
          <div ref={scrollRef} className="h-[min(540px,60vh)] overflow-auto">
            {nodes.length === 0 ? <div className="p-8 text-center text-sm text-txt-mute">Run a graph or select a completed run to see performance.</div> : null}
            <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
              {virtualizer.getVirtualItems().map((item) => {
                const node = nodes[item.index]!;
                return <PerformanceRow key={node.node_id} node={node} top={item.start} onSelect={() => selectNode(node.node_id)} />;
              })}
            </div>
          </div>
        </div>
      </div>
      <footer className="flex flex-wrap gap-x-3 border-t border-line px-3 py-1.5 font-mono text-[9px] text-txt-mute">
        Rates are wall-clock flow. Service cap is execute-time capacity. Violet is resource wait; other heat is neutral.
      </footer>
    </section>
  );
}

function PerformanceRow({ node, top, onSelect }: { node: NodeRunSnapshot; top: number; onSelect: () => void }) {
  const metrics = node.performance;
  return (
    <button
      type="button"
      className="absolute left-0 grid w-full grid-cols-[minmax(150px,1.4fr)_repeat(8,minmax(82px,0.72fr))_88px] items-center border-b border-line px-3 py-2 font-mono text-[10px] hover:bg-slate-50"
      style={{ height: 70, transform: `translateY(${top}px)` }}
      onClick={onSelect}
    >
      <div className="min-w-0 pr-3 text-left">
        <strong className="block truncate text-[11px] text-txt">{node.node_id}</strong>
        <BatchTimeline node={node} />
      </div>
      <Value value={formatRate(metrics.arrival_rate)} />
      <Value value={formatRate(metrics.departure_rate)} />
      <Value value={`${metrics.queue_size}/${metrics.queue_capacity || "—"} · ${(metrics.queue_fill_ratio * 100).toFixed(0)}%`} />
      <Value value={`${metrics.queue_growth_rate >= 0 ? "+" : ""}${metrics.queue_growth_rate.toFixed(1)}`} />
      <Value value={`${(metrics.busy_ratio * 100).toFixed(0)}%`} />
      <Value value={`${(metrics.resource_wait_ratio * 100).toFixed(0)}%`} resource />
      <Value value={formatDuration(metrics.downstream_blocked_ms)} />
      <Value value={formatDuration(metrics.batch_p95_ms)} />
      <Value value={formatRate(metrics.service_capacity)} />
    </button>
  );
}

function BatchTimeline({ node }: { node: NodeRunSnapshot }) {
  const recent = node.performance.recent_batches.slice(-12);
  return (
    <div className="mt-1 flex h-3 gap-px" title="Latest batches: queue, resource, load, execute, route">
      {recent.map((batch) => {
        const total = batch.queue_wait_ms + batch.resource_wait_ms + batch.load_ms + batch.execute_ms + batch.unload_ms + batch.route_ms || 1;
        return (
          <div key={batch.batch_index} className="flex min-w-[5px] flex-1 overflow-hidden rounded-sm" title={`batch ${batch.batch_index} · ${formatDuration(batch.total_ms)}`}>
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

function Value({ value, resource = false }: { value: string; resource?: boolean }) {
  return <span className={`text-right ${resource ? "text-violet-700" : "text-txt-dim"}`}>{value}</span>;
}

function orderedNodes(nodes: NodeRunSnapshot[], graphOrder: string[], sort: SortState): NodeRunSnapshot[] {
  const rows = [...nodes];
  if (sort) {
    const direction = sort.ascending ? 1 : -1;
    rows.sort((left, right) => (performanceSortValue(left, sort.key) - performanceSortValue(right, sort.key)) * direction);
    return rows;
  }
  const order = new Map(graphOrder.map((nodeId, index) => [nodeId, index]));
  rows.sort((left, right) => (order.get(left.node_id) ?? Number.MAX_SAFE_INTEGER) - (order.get(right.node_id) ?? Number.MAX_SAFE_INTEGER));
  return rows;
}
