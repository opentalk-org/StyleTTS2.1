import { useEffect, useState } from "react";

import { fetchJobGraph, type Job } from "@/features/jobs/api";
import { useJobsQuery } from "@/features/jobs/query";
import { Icon } from "@/shared/icons";
import { fmtAgo } from "@/shared/format";
import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { Button } from "@/shared/ui/Button";
import { IconButton } from "@/shared/ui/IconButton";
import { Input } from "@/shared/ui/Input";
import { fetchRunSnapshot } from "../api";
import { runtimeConfigForGraph, workflowDefinition } from "../logic";
import { useDeleteWorkflowMutation, useExampleWorkflowsQuery, useSaveWorkflowMutation, useSavedWorkflowsQuery } from "../query";
import { useWorkflowStore } from "../store";
import type { WorkflowGraph } from "../types";

type LibraryTab = "examples" | "saved" | "runs";

const STATE_LABEL: Record<Job["state"], string> = {
  queued: "queued",
  running: "running",
  stopping: "stopping",
  stopped: "stopped",
  succeeded: "done",
  failed: "error",
};

const STATE_CLASS: Record<Job["state"], string> = {
  queued: "border-blue-200 bg-blue-50 text-blue-700",
  running: "border-blue-200 bg-blue-50 text-blue-700",
  stopping: "border-amber-200 bg-amber-50 text-amber-700",
  stopped: "border-amber-200 bg-amber-50 text-amber-700",
  succeeded: "border-emerald-200 bg-emerald-50 text-emerald-700",
  failed: "border-red-200 bg-red-50 text-red-700",
};

export function WorkflowLibraryPopover({ onClose }: { onClose: () => void }) {
  const { schema, graph, runtimeConfig, setGraph, setRuntimeConfig, setActiveRunId, applyRunSnapshot } = useWorkflowStore();
  const workflows = useSavedWorkflowsQuery();
  const examples = useExampleWorkflowsQuery();
  const jobs = useJobsQuery({ limit: 100, offset: 0 });
  const saveWorkflow = useSaveWorkflowMutation();
  const deleteWorkflow = useDeleteWorkflowMutation();
  const [tab, setTab] = useState<LibraryTab>("examples");
  const [name, setName] = useState(`workflow_${new Date().toISOString().slice(0, 16).replace("T", "_")}`);
  const [openingRunId, setOpeningRunId] = useState<string | null>(null);
  useEffect(() => {
    const onKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [onClose]);
  if (!schema) return null;
  const loadGraph = (nextGraph: WorkflowGraph, nextConfig = runtimeConfig) => {
    setGraph({ nodes: nextGraph.nodes, edges: nextGraph.edges, panels: nextGraph.panels ?? [] });
    setRuntimeConfig(runtimeConfigForGraph(schema, nextGraph, nextConfig));
    setActiveRunId(null);
    onClose();
  };
  const loadRun = async (job: Job) => {
    setOpeningRunId(job.run_id);
    try {
      const [runGraph, snapshot] = await Promise.all([
        fetchJobGraph(job.run_id),
        fetchRunSnapshot(job.run_id).then(
          (result) => result,
          (error) => {
            console.warn(`Run snapshot unavailable for ${job.run_id}`, error);
            return null;
          },
        ),
      ]);
      setGraph({ nodes: runGraph.nodes, edges: runGraph.edges });
      setRuntimeConfig(runtimeConfigForGraph(schema, runGraph, runGraph.context.config));
      setActiveRunId(job.run_id);
      if (snapshot) applyRunSnapshot(job.run_id, snapshot);
      onClose();
    } finally {
      setOpeningRunId(null);
    }
  };
  const savedConfig = runtimeConfigForGraph(schema, graph, runtimeConfig);
  const jobRows = jobs.data?.rows ?? [];
  return (
    <div
      className="fixed inset-0 z-[300] flex items-center justify-center bg-gray-900/45 p-6"
      role="dialog"
      aria-modal="true"
      aria-label="Workflow library"
      onClick={onClose}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <section
        className="grid max-h-[86vh] w-[min(920px,calc(100vw-32px))] grid-rows-[auto_auto_minmax(0,1fr)] overflow-hidden rounded-lg border border-line bg-panel shadow-[0_26px_90px_rgba(17,24,39,0.26)]"
        onClick={(event) => event.stopPropagation()}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-emerald-50 text-emerald-700">
                <Icon name="folder-open" size={17} strokeWidth={2.4} />
              </span>
              <div className="min-w-0">
                <h2 className="text-[18px] font-bold text-txt">Workflow library</h2>
                <p className="truncate text-[12px] text-txt-mute">Load examples, saved workflows, or previous runs.</p>
              </div>
            </div>
          </div>
          <button
            type="button"
            className="flex h-8 w-8 flex-none cursor-pointer items-center justify-center rounded-md text-txt-mute hover:bg-panel-2 hover:text-txt"
            onClick={onClose}
            aria-label="Close workflow library"
          >
            <Icon name="x" size={17} strokeWidth={2.4} />
          </button>
        </header>

        <div className="grid gap-3 border-b border-line px-5 py-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
          <div className="flex min-w-0 gap-1 rounded-md bg-panel-2 p-1">
            <TabButton active={tab === "examples"} label="Examples" count={examples.data?.length ?? 0} onClick={() => setTab("examples")} />
            <TabButton active={tab === "saved"} label="Saved" count={workflows.data?.length ?? 0} onClick={() => setTab("saved")} />
            <TabButton active={tab === "runs"} label="Runs" count={jobs.data?.total ?? jobRows.length} onClick={() => setTab("runs")} />
          </div>
          <div className="flex min-w-0 items-center gap-2">
            <Input filled className="h-9 min-w-0 flex-1 font-mono lg:w-72 lg:flex-none" value={name} onChange={(event) => setName(event.target.value)} />
            <Button
              size="sm"
              variant="primary"
              icon="download"
              disabled={!name.trim() || saveWorkflow.isPending}
              onClick={() => saveWorkflow.mutate({ name: name.trim(), data: workflowDefinition(graph, savedConfig), hidden: false })}
            >
              Save
            </Button>
          </div>
        </div>

        <div className="min-h-0 overflow-y-auto p-5">
          {tab === "examples" ? (
            examples.data?.length ? (
              <div className="grid gap-2 md:grid-cols-2">
                {examples.data.map((example) => (
                  <WorkflowRow
                    key={example.id}
                    title={example.name}
                    detail={`${example.data.nodes.length} nodes / ${example.data.edges.length} edges`}
                    meta="example"
                    icon="sparkles"
                    onClick={() => loadGraph(example.data, example.data.context.config)}
                  />
                ))}
              </div>
            ) : (
              <EmptyLibraryState text={examples.isLoading ? "Loading examples..." : "No example workflows found."} />
            )
          ) : null}

          {tab === "saved" ? (
            workflows.data?.length ? (
              <div className="grid gap-2 md:grid-cols-2">
                {workflows.data.map((workflow) => (
                  <WorkflowRow
                    key={workflow.id}
                    title={workflow.name}
                    detail={`${workflow.data.nodes.length} nodes / ${workflow.data.edges.length} edges`}
                    meta={workflow.hidden ? "hidden" : "saved"}
                    icon="folder-open"
                    onClick={() => loadGraph(workflow.data, workflow.data.context.config)}
                    onDelete={() =>
                      askConfirm({
                        title: "Delete workflow?",
                        desc: `Remove "${workflow.name}" from your saved workflows. This cannot be undone.`,
                        danger: true,
                        label: "Delete workflow",
                        onConfirm: () => deleteWorkflow.mutate({ id: workflow.id, name: workflow.name }),
                      })
                    }
                  />
                ))}
              </div>
            ) : (
              <EmptyLibraryState text={workflows.isLoading ? "Loading saved workflows..." : "No saved workflows yet."} />
            )
          ) : null}

          {tab === "runs" ? (
            jobs.isError ? (
              <EmptyLibraryState text="Could not load run history." />
            ) : jobRows.length ? (
              <div className="grid gap-2">
                {jobRows.map((job) => (
                  <RunRow key={job.run_id} job={job} opening={openingRunId === job.run_id} onClick={() => loadRun(job)} />
                ))}
              </div>
            ) : (
              <EmptyLibraryState text={jobs.isLoading ? "Loading run history..." : "No workflow runs yet."} />
            )
          ) : null}
        </div>
      </section>
    </div>
  );
}

function TabButton({ active, label, count, onClick }: { active: boolean; label: string; count: number; onClick: () => void }) {
  return (
    <button
      type="button"
      className={`flex min-w-0 flex-1 cursor-pointer items-center justify-center gap-2 rounded px-3 py-1.5 text-[12px] font-semibold ${active ? "bg-panel text-txt shadow-sm" : "text-txt-mute hover:text-txt"}`}
      onClick={onClick}
    >
      <span className="truncate">{label}</span>
      <span className="font-mono text-[10px] text-txt-mute">{count.toLocaleString()}</span>
    </button>
  );
}

function WorkflowRow({ title, detail, meta, icon, onClick, onDelete }: { title: string; detail: string; meta: string; icon: "folder-open" | "sparkles"; onClick: () => void; onDelete?: () => void }) {
  return (
    <div className="group relative">
      <button type="button" className={`grid min-h-20 w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)] gap-3 rounded-md border border-line px-3 py-3 text-left hover:bg-panel-2 ${onDelete ? "pr-11" : ""}`} onClick={onClick}>
        <Icon name={icon} size={15} className="text-txt-mute" />
        <span className="min-w-0 flex-1">
          <strong className="block truncate text-[13px] text-txt">{title}</strong>
          <span className="mt-0.5 line-clamp-2 text-[11px] leading-4 text-txt-mute">{detail}</span>
          <span className="mt-2 block font-mono text-[10px] uppercase text-txt-mute">{meta}</span>
        </span>
      </button>
      {onDelete ? (
        <IconButton
          icon="trash"
          danger
          title="Delete workflow"
          className="absolute right-2 top-2 opacity-0 group-hover:opacity-100"
          onClick={(event) => {
            event.stopPropagation();
            onDelete();
          }}
        />
      ) : null}
    </div>
  );
}

function RunRow({ job, opening, onClick }: { job: Job; opening: boolean; onClick: () => void }) {
  const updatedAt = Date.parse(job.updated_at);
  return (
    <button
      type="button"
      className="grid min-h-16 w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 rounded-md border border-line px-3 py-3 text-left hover:bg-panel-2"
      onClick={onClick}
      disabled={opening}
    >
      <Icon name={opening ? "loader" : "play-circle"} size={16} className={opening ? "animate-spin text-blue-600" : "text-txt-mute"} />
      <span className="min-w-0">
        <span className="block truncate font-mono text-[13px] font-semibold text-txt">{job.name}</span>
        <span className="block truncate font-mono text-[11px] text-txt-mute">{job.run_id}</span>
      </span>
      <span className="flex items-center gap-2">
        <span className="hidden text-[11px] text-txt-mute sm:inline">{Number.isNaN(updatedAt) ? "-" : fmtAgo(updatedAt)}</span>
        <span className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${STATE_CLASS[job.state]}`}>{STATE_LABEL[job.state]}</span>
      </span>
    </button>
  );
}

function EmptyLibraryState({ text }: { text: string }) {
  return <p className="rounded-md border border-line px-4 py-8 text-center font-mono text-[12px] text-txt-mute">{text}</p>;
}
