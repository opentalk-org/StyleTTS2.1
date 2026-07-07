import { useState } from "react";

import { useNav } from "@/app/navStore";
import { fetchRunSnapshot, fetchWorkflowSchema } from "@/features/workflows/api";
import { useWorkflowStore } from "@/features/workflows/store";
import { Pager } from "@/shared/data/Pager";
import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { fmtAgo } from "@/shared/format";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { IconButton } from "@/shared/ui/IconButton";
import { Select } from "@/shared/ui/Select";
import { fetchJobGraph, type Job } from "./api";
import { useJobActions, useJobsQuery } from "./query";

const STATE_LABEL: Record<Job["state"], string> = {
  queued: "queued",
  running: "running",
  stopping: "stopping",
  stopped: "stopped",
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
  const [editing, setEditing] = useState(false);
  const { remove, removing, stop, stopping, rename } = useJobActions();
  const active = ACTIVE_STATES.has(job.state);
  const commitRename = (value: string) => {
    setEditing(false);
    const next = value.trim();
    if (next && next !== job.name) rename(job.run_id, next);
  };
  const openJob = async () => {
    setOpening(true);
    try {
      const [schema, graph] = await Promise.all([fetchWorkflowSchema(), fetchJobGraph(job.run_id)]);
      const workflow = useWorkflowStore.getState();
      workflow.setSchema(schema);
      workflow.setGraph({ nodes: graph.nodes, edges: graph.edges });
      workflow.setRuntimeConfig(graph.context.config);
      workflow.setActiveRunId(job.run_id);
      const snapshot = await fetchRunSnapshot(job.run_id).catch(() => null);
      if (snapshot) workflow.applyRunSnapshot(job.run_id, snapshot);
      useNav.getState().go("workflows");
    } finally {
      setOpening(false);
    }
  };
  const removeJob = () =>
    askConfirm({
      title: active ? "Stop and remove job?" : "Remove job?",
      desc: active
        ? `"${job.name}" is still running. Removing it stops the run and prunes its cached node logs. This cannot be undone.`
        : `Remove "${job.name}" and prune its cached node logs. This cannot be undone.`,
      danger: true,
      label: active ? "Stop and remove" : "Remove job",
      onConfirm: () => remove(job.run_id),
    });
  const updatedAt = Date.parse(job.updated_at);
  return (
    <div className="grid items-center gap-3 border-b border-line px-4 py-3 last:border-b-0" style={{ gridTemplateColumns: "110px minmax(180px,1fr) 150px 180px" }}>
      <span className={job.state === "failed" ? "text-red-600" : job.state === "succeeded" ? "text-emerald-700" : job.state === "stopped" ? "text-amber-700" : "text-blue-700"}>
        {STATE_LABEL[job.state]}
      </span>
      <div className="min-w-0">
        {editing ? (
          <input
            defaultValue={job.name}
            autoFocus
            onFocus={(e) => e.target.select()}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename((e.target as HTMLInputElement).value);
              if (e.key === "Escape") setEditing(false);
            }}
            onBlur={(e) => commitRename(e.target.value)}
            className="h-[26px] w-full max-w-[320px] rounded-md border-2 border-blue-500 bg-panel-2 px-2 font-mono text-[13px] font-semibold text-txt outline-none"
          />
        ) : (
          <button
            onDoubleClick={() => setEditing(true)}
            title="Double-click to rename"
            className="block max-w-full cursor-text truncate text-left font-mono text-[13px] font-semibold text-txt"
          >
            {job.name}
          </button>
        )}
        <div className="truncate font-mono text-[11px] text-txt-mute">{job.run_id}</div>
      </div>
      <span className="text-xs text-txt-mute">{Number.isNaN(updatedAt) ? "-" : fmtAgo(updatedAt)}</span>
      <div className="flex justify-end gap-2">
        <IconButton icon="edit" title="Rename" onClick={() => setEditing((v) => !v)} />
        <Button variant="secondary" size="sm" icon="folder-open" onClick={openJob} disabled={opening}>
          Open
        </Button>
        {active ? (
          <Button variant="secondary" size="sm" icon="x" onClick={() => stop(job.run_id)} disabled={stopping || job.state === "stopping"}>
            Stop
          </Button>
        ) : null}
        <Button variant="danger" size="sm" icon="trash" onClick={removeJob} disabled={removing}>
          Remove
        </Button>
      </div>
    </div>
  );
}
