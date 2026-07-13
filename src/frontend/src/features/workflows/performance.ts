import type { EdgePerformanceSnapshot, NodeRunSnapshot, RunSnapshot, WorkflowEdge } from "./types";

export type PerformanceSortKey = "arrival" | "departure" | "queueFill" | "queueGrowth" | "busy" | "resourceWait" | "blocked" | "p95";

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

export function neutralQueueTone(fillRatio: number): string {
  const alpha = Math.min(0.18, Math.max(0, fillRatio) * 0.18);
  return `rgba(100, 116, 139, ${alpha.toFixed(3)})`;
}

export function flowStrokeWidth(rate: number, maximumRate: number): number {
  if (maximumRate <= 0 || rate <= 0) return 1.5;
  return 1.5 + Math.sqrt(rate / maximumRate) * 4.5;
}

export function performanceSortValue(node: NodeRunSnapshot, key: PerformanceSortKey): number {
  const metrics = node.performance;
  if (key === "arrival") return metrics.arrival_rate;
  if (key === "departure") return metrics.departure_rate;
  if (key === "queueFill") return metrics.queue_fill_ratio;
  if (key === "queueGrowth") return metrics.queue_growth_rate;
  if (key === "busy") return metrics.busy_ratio;
  if (key === "resourceWait") return metrics.resource_wait_ratio;
  if (key === "blocked") return metrics.downstream_blocked_ms;
  return metrics.batch_p95_ms;
}
