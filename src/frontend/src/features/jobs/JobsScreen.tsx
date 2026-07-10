import { useState } from "react";

import { useNav } from "@/app/navStore";
import { fetchRunSnapshot, fetchWorkflowSchema } from "@/features/workflows/api";
import { useWorkflowStore } from "@/features/workflows/store";
import { Pager } from "@/shared/data/Pager";
import { VirtualTable } from "@/shared/data/VirtualTable";
import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { fmtAgo } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { IconButton } from "@/shared/ui/IconButton";
import { Select } from "@/shared/ui/Select";
import { cn } from "@/shared/ui/cn";
import { fetchJobGraph, type Job } from "./api";
import { useJobActions, useJobsQuery } from "./query";
import { useJobsSelection } from "./store";

const STATE_LABEL: Record<Job["state"], string> = {
  queued: "queued",
  running: "running",
  stopping: "stopping",
  stopped: "stopped",
  succeeded: "done",
  failed: "error",
};

const ACTIVE_STATES = new Set<Job["state"]>(["running", "stopping"]);
const JOB_COLS = "32px 110px minmax(180px,1fr) 150px 180px";

function Checkbox({ on }: { on: boolean }) {
  return (
    <span className={cn("flex h-[18px] w-[18px] flex-none items-center justify-center rounded", on ? "bg-blue-500" : "border-2 border-line-2 bg-panel")}>
      {on ? <Icon name="check" size={12} strokeWidth={3} className="text-white" /> : null}
    </span>
  );
}

export function JobsScreen() {
  const [limit, setLimit] = useState(100);
  const [offset, setOffset] = useState(0);
  const jobs = useJobsQuery({ limit, offset });
  const { removeMany, removeAll, removingMany, removingAll } = useJobActions();
  const { selection, toggleSelect, selectMany, clearSelection } = useJobsSelection();
  const rows = jobs.data?.rows ?? [];
  const total = jobs.data?.total ?? 0;
  const page = Math.floor(offset / limit);
  const pages = Math.max(1, Math.ceil(total / limit));
  const visibleEnd = Math.min(offset + rows.length, total);
  const selectedIds = Object.keys(selection);
  const allSel = rows.length > 0 && rows.every((job) => selection[job.run_id]);

  const removeSelected = () =>
    askConfirm({
      title: "Remove jobs?",
      desc: `Remove ${selectedIds.length} selected job${selectedIds.length === 1 ? "" : "s"}, stopping any still running and pruning their cached node logs. This cannot be undone.`,
      danger: true,
      label: "Remove jobs",
      onConfirm: () => removeMany(selectedIds, { onSuccess: () => clearSelection() }),
    });
  const removeEverything = () =>
    askConfirm({
      title: "Remove all jobs?",
      desc: `Remove all ${total.toLocaleString()} jobs, stopping any still running and pruning their cached node logs. This cannot be undone.`,
      danger: true,
      label: "Remove all jobs",
      onConfirm: () => removeAll({ onSuccess: () => clearSelection() }),
    });

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
        <div className="flex-1" />
        <Button variant="danger" size="sm" icon="trash" onClick={removeEverything} disabled={removingAll || total === 0}>
          Remove all
        </Button>
      </div>
      {jobs.isLoading ? (
        <Card className="p-6 text-sm text-txt-mute">Loading jobs...</Card>
      ) : jobs.isError ? (
        <Card>
          <EmptyState icon="alert" title="Couldn't reach the backend" description="The jobs service didn't respond." />
        </Card>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          {selectedIds.length ? (
            <div className="mb-3 flex items-center gap-2.5 rounded-[9px] bg-gray-900 py-2.5 pl-4 pr-3">
              <span className="text-[13px] font-bold text-white">{selectedIds.length.toLocaleString()} selected</span>
              <div className="flex-1" />
              <button
                disabled={removingMany}
                onClick={removeSelected}
                className="flex h-[34px] items-center gap-1.5 rounded-md bg-red-500 px-3 text-[12.5px] font-semibold text-white hover:bg-red-600 disabled:cursor-default disabled:opacity-60"
              >
                <Icon name="trash" size={15} strokeWidth={2.2} />
                Remove
              </button>
              <button
                onClick={clearSelection}
                title="Clear selection"
                className="flex h-[34px] w-[34px] items-center justify-center rounded-md bg-white/10 text-white hover:bg-white/15"
              >
                <Icon name="x" size={16} strokeWidth={2.4} />
              </button>
            </div>
          ) : null}
          <div className="mb-2.5 flex items-center gap-3 text-xs tabular-nums text-txt-mute">
            <span>{total ? `${(offset + 1).toLocaleString()}-${visibleEnd.toLocaleString()}` : "0"} of {total.toLocaleString()} jobs</span>
            <Pager page={page} pages={pages} onChange={(next) => setOffset(next * limit)} />
          </div>
          {rows.length ? (
            <Card className="min-h-0 flex-1 overflow-hidden">
              <VirtualTable
                count={rows.length}
                estimateRowHeight={64}
                className="h-full"
                header={
                  <div className="grid h-10 flex-none items-center gap-3 border-b border-line px-4" style={{ gridTemplateColumns: JOB_COLS }}>
                    <button onClick={() => (allSel ? clearSelection() : selectMany(rows.map((job) => job.run_id)))} className="flex">
                      <Checkbox on={allSel} />
                    </button>
                    <span className="text-[11px] font-bold uppercase tracking-wider text-txt-mute">State</span>
                    <span className="text-[11px] font-bold uppercase tracking-wider text-txt-mute">Job</span>
                    <span className="text-[11px] font-bold uppercase tracking-wider text-txt-mute">Updated</span>
                    <span />
                  </div>
                }
                renderRow={(index) => {
                  const job = rows[index]!;
                  return <JobRow job={job} selected={!!selection[job.run_id]} onToggle={() => toggleSelect(job.run_id)} />;
                }}
              />
            </Card>
          ) : (
            <Card>
              <EmptyState icon="list-checks" title="No workflow jobs have run yet." />
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

function JobRow({ job, selected, onToggle }: { job: Job; selected: boolean; onToggle: () => void }) {
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
    <div className={cn("grid items-center gap-3 border-b border-line px-4 py-3 last:border-b-0", selected && "bg-blue-50")} style={{ gridTemplateColumns: JOB_COLS }}>
      <button onClick={onToggle} className="flex">
        <Checkbox on={selected} />
      </button>
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
