import { useEffect, useState } from "react";

import { useRunnersQuery } from "@/features/cluster/query";
import { Button } from "@/shared/ui/Button";
import { Select } from "@/shared/ui/Select";
import { defaultWorkflowContext, graphPayload, runtimeConfigForGraph } from "../logic";
import { useRunSnapshotQuery, useStartGraphMutation, useStopRunMutation } from "../query";
import { useWorkflowStore } from "../store";

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
    <div className="absolute right-4 top-4 z-10 flex max-w-[520px] items-center gap-2 rounded-md border border-line bg-panel p-2 shadow-lg">
      <span className="px-2 text-[12px] font-semibold text-txt-dim">
        {active ? `${active.run_id} · ${active.state} · ${snapshot?.total_event_count ?? active.event_count} events` : `${graph.nodes.length} nodes`}
      </span>
      {canStop ? (
        <Button variant="secondary" icon="pause" onClick={() => stop.mutate(active.run_id)}>Stop</Button>
      ) : (
        <>
          <Select className="w-[180px]" variant="mini" value={runnerId} onChange={setRunnerId} options={runnerOptions} />
          <Button variant="primary" icon="play" disabled={!runnerId || start.isPending} onClick={runGraph}>Run</Button>
        </>
      )}
    </div>
  );
}
