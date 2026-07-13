import { formatRate, neutralQueueTone } from "../performance";
import type { NodePerformanceSnapshot } from "../types";

export function NodePerformanceSummary({ performance }: { performance: NodePerformanceSnapshot }) {
  const observed = performance.arrived_items > 0 || performance.departed_items > 0 || performance.current_batch_started_at !== null;
  const growth = performance.queue_growth_rate;
  const growthText = growth === 0 ? "" : `${growth > 0 ? "↑" : "↓"}${formatRate(Math.abs(growth))}`;
  const busyWidth = `${performance.busy_ratio * 100}%`;
  const resourceWidth = `${performance.busy_ratio * performance.resource_wait_ratio * 100}%`;
  return (
    <div
      className="relative flex h-8 items-center gap-2 overflow-hidden border-b border-line px-2.5 font-mono text-[9px] text-txt-dim"
      style={{ backgroundColor: neutralQueueTone(performance.queue_fill_ratio) }}
      title={`Busy ${(performance.busy_ratio * 100).toFixed(0)}% · resource wait ${(performance.resource_wait_ratio * 100).toFixed(0)}%`}
    >
      <strong className="whitespace-nowrap text-[10px] text-txt">
        {observed ? `${formatRate(performance.arrival_rate)} → ${formatRate(performance.departure_rate)}` : "— → —"}
      </strong>
      <span className="ml-auto whitespace-nowrap rounded bg-white/70 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wide text-txt-mute">
        q {performance.queue_size}/{performance.queue_capacity || "—"}{growthText ? ` ${growthText}` : ""}
      </span>
      <i className="absolute bottom-0 left-0 h-0.5 bg-slate-400" style={{ width: busyWidth }} />
      <i className="absolute bottom-0 left-0 h-0.5 bg-violet-500" style={{ width: resourceWidth }} />
    </div>
  );
}
