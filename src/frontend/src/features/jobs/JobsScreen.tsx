import { useState } from "react";

import { useNav } from "@/app/navStore";
import { fetchWorkflowSchema } from "@/features/workflows/api";
import { useWorkflowStore } from "@/features/workflows/store";
import { Pager } from "@/shared/data/Pager";
import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { fmtAgo } from "@/shared/format";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { Select } from "@/shared/ui/Select";
import { fetchJobGraph, type Job } from "./api";
import { useJobActions, useJobsQuery } from "./query";

const STATE_LABEL: Record<Job["state"], string> = {
  queued: "queued",
  running: "running",
  stopping: "stopping",
  stopped: "done",
  succeeded: "done",
  failed: "error",
};

const ACTIVE_STATES = new Set<Job["state"]>(["running", "stopping"]);

export function JobsScreen() {
  const [limit, setLimit] = useState(100);
  const [offset, setOffset] = useState(0);
  const jobs = useJobsQuery({ limit, offset });
  const rows = jobs.data?.rows ?? [];
  const total = jobs.data?.total ?? 0;
  const page = Math.floor(offset / limit);
  const pages = Math.max(1, Math.ceil(total / limit));
  const visibleEnd = Math.min(offset + rows.length, total);

  return (
    <div className="mx-auto flex h-full max-w-[1080px] flex-col px-7 pb-6 pt-5">
      <div className="mb-3.5 flex items-center gap-2.5">
        <Select
          variant="mini"
          value={String(limit)}
          onChange={(value) => {
            setLimit(Number(value));
            setOffset(0);
          }}
          options={[
            { value: "50", label: "50 per page" },
            { value: "100", label: "100 per page" },
            { value: "200", label: "200 per page" },
          ]}
        />
      </div>
      {jobs.isLoading ? (
        <Card className="p-6 text-sm text-txt-mute">Loading jobs...</Card>
      ) : jobs.isError ? (
        <Card>
          <EmptyState icon="alert" title="Couldn't reach the backend" description="The jobs service didn't respond." />
        </Card>
      ) : (
        <>
          <div className="mb-2.5 flex items-center gap-3 text-xs tabular-nums text-txt-mute">
            <span>{total ? `${(offset + 1).toLocaleString()}-${visibleEnd.toLocaleString()}` : "0"} of {total.toLocaleString()} jobs</span>
            <Pager page={page} pages={pages} onChange={(next) => setOffset(next * limit)} />
          </div>
          {rows.length ? (
            <Card className="overflow-hidden">
              {rows.map((job) => <JobRow key={job.run_id} job={job} />)}
            </Card>
          ) : (
            <Card>
              <EmptyState icon="list-checks" title="No workflow jobs have run yet." />
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function JobRow({ job }: { job: Job }) {
  const [opening, setOpening] = useState(false);
  const { remove, removing, stop, stopping } = useJobActions();
  const active = ACTIVE_STATES.has(job.state);
  const openJob = async () => {
    setOpening(true);
    try {
      const [schema, graph] = await Promise.all([fetchWorkflowSchema(), fetchJobGraph(job.run_id)]);
      const workflow = useWorkflowStore.getState();
      workflow.setSchema(schema);
      workflow.setGraph({ nodes: graph.nodes, edges: graph.edges });
      workflow.setRuntimeConfig(graph.context.config);
      workflow.setActiveRunId(job.run_id);
      useNav.getState().go("workflows");
    } finally {
      setOpening(false);
    }
  };
  const removeJob = () =>
    askConfirm({
      title: "Remove job?",
      desc: `Remove "${job.name}" and prune its cached node logs. This cannot be undone.`,
      danger: true,
      label: "Remove job",
      onConfirm: () => remove(job.run_id),
    });
  const updatedAt = Date.parse(job.updated_at);
  return (
    <div className="grid items-center gap-3 border-b border-line px-4 py-3 last:border-b-0" style={{ gridTemplateColumns: "110px minmax(180px,1fr) 150px 180px" }}>
      <span className={job.state === "failed" ? "text-red-600" : job.state === "succeeded" || job.state === "stopped" ? "text-emerald-700" : "text-blue-700"}>
        {STATE_LABEL[job.state]}
      </span>
      <div className="min-w-0">
        <div className="truncate font-mono text-[13px] font-semibold text-txt">{job.name}</div>
        <div className="truncate font-mono text-[11px] text-txt-mute">{job.run_id}</div>
      </div>
      <span className="text-xs text-txt-mute">{Number.isNaN(updatedAt) ? "-" : fmtAgo(updatedAt)}</span>
      <div className="flex justify-end gap-2">
        <Button variant="secondary" size="sm" icon="folder-open" onClick={openJob} disabled={opening}>
          Open
        </Button>
        {active ? (
          <Button variant="danger" size="sm" icon="x" onClick={() => stop(job.run_id)} disabled={stopping || job.state === "stopping"}>
            Stop
          </Button>
        ) : (
          <Button variant="danger" size="sm" icon="trash" onClick={removeJob} disabled={removing}>
            Remove
          </Button>
        )}
      </div>
    </div>
  );
}
