import { formatDuration, formatRate, formatResourceWait, neutralQueueTone } from "../performance";
import type { NodePerformanceSnapshot } from "../types";

export function NodePerformanceSummary({ performance }: { performance: NodePerformanceSnapshot }) {
  const observed = performance.arrived_items > 0 || performance.departed_items > 0 || performance.current_batch_started_at !== null;
  const busyWidth = `${performance.busy_ratio * 100}%`;
  const resourceWidth = `${performance.busy_ratio * performance.resource_wait_ratio * 100}%`;
  return (
    <div
      className="relative flex h-11 items-center gap-2 overflow-hidden border-b border-line px-2.5 font-mono text-[9px] text-txt-dim"
      style={{ backgroundColor: neutralQueueTone(performance.queue_fill_ratio) }}
      title={`Busy ${(performance.busy_ratio * 100).toFixed(0)}% · p95 batch ${formatDuration(performance.batch_p95_ms)}`}
    >
      <div className="grid grid-cols-2 gap-1">
        <Rate label="in" value={observed ? formatRate(performance.arrival_rate) : "—"} />
        <Rate label="out" value={observed ? formatRate(performance.departure_rate) : "—"} />
      </div>
      <div className="ml-auto grid gap-0.5 text-right">
        <Wait label="resource wait" value={formatResourceWait(performance)} tone="resource" />
        <Wait label="output wait" value={formatDuration(performance.downstream_blocked_ms)} tone="output" />
      </div>
      <i className="absolute bottom-0 left-0 h-0.5 bg-slate-400" style={{ width: busyWidth }} />
      <i className="absolute bottom-0 left-0 h-0.5 bg-violet-500" style={{ width: resourceWidth }} />
    </div>
  );
}

function Wait({ label, value, tone }: { label: string; value: string; tone: "resource" | "output" }) {
  return (
    <span className={`whitespace-nowrap text-[8px] font-semibold uppercase tracking-wide ${tone === "resource" ? "text-violet-700" : "text-amber-700"}`}>
      {label} <strong>{value}</strong>
    </span>
  );
}

function Rate({ label, value }: { label: string; value: string }) {
  return (
    <span className="grid min-w-[43px] text-center leading-none">
      <strong className="text-[10px] text-txt">{value}</strong>
      <small className="mt-0.5 text-[7px] font-bold uppercase text-txt-mute">{label}</small>
    </span>
  );
}
