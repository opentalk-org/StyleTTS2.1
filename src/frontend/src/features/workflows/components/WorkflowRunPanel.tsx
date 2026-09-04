import { useEffect, useState } from "react";

import { useRunnersQuery } from "@/features/cluster/query";
import { Button } from "@/shared/ui/Button";
import { Select } from "@/shared/ui/Select";
import { defaultWorkflowContext, graphPayload, runtimeConfigForGraph } from "../execution";
import { useRunSnapshotQuery, useStartGraphMutation, useStopRunMutation } from "../query";
import { useWorkflowStore } from "../store";
import { RunPerformanceStrip } from "./RunPerformanceStrip";

export function WorkflowRunPanel() {
  const { schema, graph, runtimeConfig, activeRunId, runs, snapshots, setActiveRunId, applyRunSnapshot } = useWorkflowStore();
  const runners = useRunnersQuery();
  const [runnerId, setRunnerId] = useState("");
  const start = useStartGraphMutation();
  const stop = useStopRunMutation();
  const active = runs.find((run) => run.run_id === activeRunId);
  const snapshot = activeRunId ? snapshots[activeRunId] : undefined;
  const snapshotQuery = useRunSnapshotQuery(activeRunId);
  const onlineRunners = runners.data?.rows.filter((runner) => runner.online) ?? [];
  useEffect(() => {
    if (runnerId && onlineRunners.some((runner) => runner.name === runnerId)) return;
    setRunnerId(onlineRunners[0]?.name ?? "");
  }, [onlineRunners, runnerId]);
  useEffect(() => {
    if (activeRunId && snapshotQuery.data) applyRunSnapshot(activeRunId, snapshotQuery.data);
  }, [activeRunId, applyRunSnapshot, snapshotQuery.data]);
  const canStop = active && ["queued", "running", "stopping"].includes(active.state);
  const context = defaultWorkflowContext(schema ? runtimeConfigForGraph(schema, graph, runtimeConfig) : runtimeConfig);
  const runGraph = () => start.mutate(graphPayload(graph, null, context, runnerId), { onSuccess: (run) => setActiveRunId(run.run_id) });
  const runnerOptions = onlineRunners.length
    ? onlineRunners.map((runner) => ({ value: runner.name, label: runner.busy ? `${runner.name} (busy)` : runner.name }))
    : [{ value: "", label: runners.isLoading ? "Loading runners" : "No online runners" }];

  return (
    <div className="absolute right-4 top-4 z-10 flex h-11 max-w-[calc(100vw-2rem)] items-center gap-2 rounded-lg border border-line bg-panel px-2 shadow-lg">
      {active ? <RunState state={active.state} /> : <span className="px-1 font-mono text-[10px] text-txt-mute">{graph.nodes.length} nodes</span>}
      {active ? <span className="max-w-32 truncate font-mono text-[9px] text-txt-mute" title={active.run_id}>{active.run_id}</span> : null}
      <RunPerformanceStrip snapshot={snapshot} compact />
      {canStop ? (
        <Button variant="secondary" icon="pause" onClick={() => stop.mutate(active.run_id)}>Stop</Button>
      ) : (
        <>
          <Select className="w-[150px]" variant="mini" value={runnerId} onChange={setRunnerId} options={runnerOptions} />
          <Button variant="primary" icon="play" disabled={!runnerId || start.isPending} onClick={runGraph}>Run</Button>
        </>
      )}
    </div>
  );
}

function RunState({ state }: { state: string }) {
  const tone = state === "failed" ? "bg-red-100 text-red-700" : state === "running" ? "bg-emerald-100 text-emerald-700" : state === "stopping" ? "bg-amber-100 text-amber-700" : "bg-slate-100 text-slate-600";
  return <span className={`rounded px-1.5 py-1 font-mono text-[8px] font-extrabold uppercase tracking-wide ${tone}`}>{state}</span>;
}
