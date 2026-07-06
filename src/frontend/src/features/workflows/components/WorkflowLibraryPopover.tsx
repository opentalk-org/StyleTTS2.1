import { useState } from "react";

import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { runtimeConfigForGraph, workflowDefinition } from "../logic";
import { useSaveWorkflowMutation, useSavedWorkflowsQuery } from "../query";
import { useWorkflowStore } from "../store";
import { WORKFLOW_TEMPLATES } from "../templates";

export function WorkflowLibraryPopover({ onClose }: { onClose: () => void }) {
  const { schema, graph, runtimeConfig, setGraph, setRuntimeConfig } = useWorkflowStore();
  const workflows = useSavedWorkflowsQuery();
  const saveWorkflow = useSaveWorkflowMutation();
  const [name, setName] = useState(`workflow_${new Date().toISOString().slice(0, 16).replace("T", "_")}`);
  if (!schema) return null;
  const loadGraph = (nextGraph: typeof graph, nextConfig = runtimeConfig) => {
    setGraph({ nodes: nextGraph.nodes, edges: nextGraph.edges });
    setRuntimeConfig(runtimeConfigForGraph(schema, nextGraph, nextConfig));
    onClose();
  };
  const savedConfig = runtimeConfigForGraph(schema, graph, runtimeConfig);
  return (
    <div className="absolute bottom-14 left-4 z-20 grid max-h-[520px] w-[420px] grid-rows-[auto_auto_minmax(0,1fr)] gap-3 overflow-hidden rounded-md border border-line bg-panel p-3 shadow-xl">
      <section className="grid gap-2">
        <div className="flex items-center gap-2">
          <Input filled className="h-9 min-w-0 flex-1 font-mono" value={name} onChange={(event) => setName(event.target.value)} />
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
      </section>
      <section className="grid gap-1">
        <h3 className="px-1 font-mono text-[10px] font-bold uppercase text-txt-mute">Examples</h3>
        {WORKFLOW_TEMPLATES.map((template) => (
          <WorkflowRow
            key={template.id}
            title={template.name}
            detail={template.description}
            icon="sparkles"
            onClick={() => loadGraph(template.build(schema))}
          />
        ))}
      </section>
      <section className="min-h-0 overflow-y-auto">
        <h3 className="px-1 pb-1 font-mono text-[10px] font-bold uppercase text-txt-mute">Saved</h3>
        {workflows.data?.length ? (
          workflows.data.map((workflow) => (
            <WorkflowRow
              key={workflow.id}
              title={workflow.name}
              detail={`${workflow.data.nodes.length} nodes / ${workflow.data.edges.length} edges`}
              icon="folder-open"
              onClick={() => loadGraph(workflow.data, workflow.data.context.config)}
            />
          ))
        ) : (
          <p className="px-1 py-3 font-mono text-[12px] text-txt-mute">{workflows.isLoading ? "Loading workflows..." : "No saved workflows yet."}</p>
        )}
      </section>
    </div>
  );
}

function WorkflowRow({ title, detail, icon, onClick }: { title: string; detail: string; icon: "folder-open" | "sparkles"; onClick: () => void }) {
  return (
    <button type="button" className="flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-left hover:bg-panel-2" onClick={onClick}>
      <Icon name={icon} size={15} className="text-txt-mute" />
      <span className="min-w-0 flex-1">
        <strong className="block truncate text-[13px] text-txt">{title}</strong>
        <span className="block truncate text-[11px] text-txt-mute">{detail}</span>
      </span>
    </button>
  );
}
