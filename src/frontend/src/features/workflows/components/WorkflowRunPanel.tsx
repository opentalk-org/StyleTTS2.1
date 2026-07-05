import { useEffect } from "react";

import { Button } from "@/shared/ui/Button";
import { graphPayload } from "../logic";
import { useRunSnapshotQuery, useStartGraphMutation, useStopRunMutation } from "../query";
import { useWorkflowStore } from "../store";

export function WorkflowRunPanel() {
  const { graph, runtimeConfig, activeRunId, runs, snapshots, setActiveRunId, applyRunSnapshot } = useWorkflowStore();
  const start = useStartGraphMutation();
  const stop = useStopRunMutation();
  const active = runs.find((run) => run.run_id === activeRunId);
  const snapshot = activeRunId ? snapshots[activeRunId] : undefined;
  const snapshotQuery = useRunSnapshotQuery(activeRunId);
  useEffect(() => {
    if (activeRunId && snapshotQuery.data) applyRunSnapshot(activeRunId, snapshotQuery.data);
  }, [activeRunId, applyRunSnapshot, snapshotQuery.data]);
  const canStop = active && ["queued", "running", "stopping"].includes(active.state);
  const context = { work_dir: "work", cache_dir: "cache", output_dir: "outputs", device: "cuda", config: runtimeConfig, input_items: [] };
  const runGraph = () => start.mutate(graphPayload(graph, null, context), { onSuccess: (run) => setActiveRunId(run.run_id) });

  return (
    <div className="absolute right-4 top-4 z-10 flex max-w-[520px] items-center gap-2 rounded-md border border-line bg-panel p-2 shadow-lg">
      <span className="px-2 text-[12px] font-semibold text-txt-dim">
        {active ? `${active.run_id} · ${active.state} · ${snapshot?.total_event_count ?? active.event_count} events` : `${graph.nodes.length} nodes`}
      </span>
      {canStop ? <Button variant="secondary" icon="pause" onClick={() => stop.mutate(active.run_id)}>Stop</Button> : <Button variant="primary" icon="play" onClick={runGraph}>Run</Button>}
    </div>
  );
}
