import type { EdgePerformanceSnapshot, NodePerformanceSnapshot, NodeRunSnapshot, RunSnapshot, WorkflowEdge } from "./types";

export type PerformanceSortKey = "arrival" | "departure" | "batchSize" | "execute" | "route" | "resourceWait" | "blocked" | "p95";

export type BatchAverages = { inputItems: number; outputItems: number; executeMs: number; routeMs: number };

export function edgeKey(edge: WorkflowEdge): string {
  return `${edge.source_node}:${edge.source_port}:${edge.target_node}:${edge.target_port}`;
}

export function nodePerformance(snapshot: RunSnapshot | undefined, nodeId: string): NodeRunSnapshot | undefined {
  return snapshot?.nodes.find((node) => node.node_id === nodeId);
}

export function edgePerformance(snapshot: RunSnapshot | undefined, edge: WorkflowEdge): EdgePerformanceSnapshot | undefined {
  return snapshot?.edges.find((item) =>
    item.source_node === edge.source_node && item.source_port === edge.source_port
    && item.target_node === edge.target_node && item.target_port === edge.target_port,
  );
}

export function formatRate(rate: number): string {
  if (rate === 0) return "0/s";
  return `${rate < 10 ? rate.toFixed(1) : rate.toFixed(0)}/s`;
}

export function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds.toFixed(0)}ms`;
  if (milliseconds < 60_000) return `${(milliseconds / 1000).toFixed(1)}s`;
  return `${(milliseconds / 60_000).toFixed(1)}m`;
}

export function formatResourceWait(performance: NodePerformanceSnapshot): string {
  if (performance.resource_wait_ms != null) return formatDuration(performance.resource_wait_ms);
  return `${(performance.resource_wait_ratio * 100).toFixed(1)}%`;
}

export function neutralQueueTone(fillRatio: number): string {
  const alpha = Math.min(0.18, Math.max(0, fillRatio) * 0.18);
  return `rgba(100, 116, 139, ${alpha.toFixed(3)})`;
}

export function flowStrokeWidth(rate: number, maximumRate: number): number {
  if (maximumRate <= 0 || rate <= 0) return 1.5;
  return 1.5 + Math.sqrt(rate / maximumRate) * 4.5;
}

export function servicePerformance(node: NodeRunSnapshot): NodePerformanceSnapshot {
  const metrics = node.performance;
  if (metrics.service_capacity <= 0 || metrics.departed_items <= 0) return metrics;
  const executeSeconds = metrics.departed_items / metrics.service_capacity;
  const inputItems = node.counters.packets_received ?? 0;
  const outputItems = node.counters.packets_created ?? 0;
  return {
    ...metrics,
    arrived_items: inputItems,
    departed_items: outputItems,
    arrival_rate: inputItems / executeSeconds,
    departure_rate: outputItems / executeSeconds,
  };
}

export function batchAverages(node: NodeRunSnapshot): BatchAverages {
  const metrics = node.performance;
  const recent = metrics.recent_batches;
  let executeMs = 0;
  let routeMs = 0;
  for (const batch of recent) {
    executeMs += batch.execute_ms;
    routeMs += batch.route_ms;
  }
  return {
    inputItems: metrics.batches ? (node.counters.tasks_completed ?? 0) / metrics.batches : 0,
    outputItems: metrics.batches ? (node.counters.packets_created ?? 0) / metrics.batches : 0,
    executeMs: recent.length ? executeMs / recent.length : 0,
    routeMs: recent.length ? routeMs / recent.length : 0,
  };
}

export function performanceSortValue(node: NodeRunSnapshot, key: PerformanceSortKey): number {
  const metrics = servicePerformance(node);
  const batch = batchAverages(node);
  if (key === "arrival") return metrics.arrival_rate;
  if (key === "departure") return metrics.departure_rate;
  if (key === "batchSize") return batch.inputItems;
  if (key === "execute") return batch.executeMs;
  if (key === "route") return batch.routeMs;
  if (key === "resourceWait") return metrics.resource_wait_ms ?? metrics.resource_wait_ratio;
  if (key === "blocked") return metrics.downstream_blocked_ms;
  return metrics.batch_p95_ms;
}
