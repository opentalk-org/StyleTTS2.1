import { formatDuration, formatRate } from "../performance";
import type { RunSnapshot } from "../types";

export function RunPerformanceStrip({ snapshot, compact }: { snapshot: RunSnapshot | undefined; compact: boolean }) {
  if (!snapshot) return null;
  const performance = snapshot.performance;
  if (compact) {
    return (
      <div className="flex items-baseline gap-2 whitespace-nowrap border-l border-line pl-2 font-mono">
        <strong className="text-[13px] text-txt">{formatRate(performance.rolling_throughput)}</strong>
        <span className="text-[9px] text-txt-mute">{performance.completed_items}/{performance.started_items} · {performance.inflight_items} live</span>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-2 font-mono text-[10px] text-txt-dim">
      <strong className="text-txt">{formatRate(performance.rolling_throughput)} rolling</strong>
      <span>{formatRate(performance.average_throughput)} average</span>
      <span>{performance.completed_items}/{performance.started_items} complete</span>
      <span>{performance.inflight_items} in flight</span>
      <span>p50 {formatDuration(performance.latency_p50_ms)}</span>
      <span>p95 {formatDuration(performance.latency_p95_ms)}</span>
    </div>
  );
}
