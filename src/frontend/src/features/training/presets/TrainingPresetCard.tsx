import { useRef, useState, type ChangeEvent } from "react";

import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";

import { runtimeConfigForGraph, workflowDefinition } from "../../workflows/execution";
import type { WorkflowDefinition, WorkflowGraph, WorkflowSchema } from "../../workflows/types";
import { TRAINING_WORKFLOWS } from "../logic";
import { useCreateTrainingConfigMutation, useTrainingConfigsQuery } from "../query";
import type { TrainTab } from "../store";

const PRESET_VERSION = 1;

export function TrainingPresetCard({
  tab,
  schema,
  graph,
  onLoad,
}: {
  tab: TrainTab;
  schema: WorkflowSchema;
  graph: WorkflowGraph;
  onLoad: (graph: WorkflowGraph) => void;
}) {
  const type = `training_preset_${tab}`;
  const presets = useTrainingConfigsQuery(type);
  const createPreset = useCreateTrainingConfigMutation(type);
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("");
  const [fileError, setFileError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const selected = presets.data?.find((preset) => preset.id === selectedId);
  const options = [
    { value: "", label: presets.isLoading ? "Loading presets…" : "— select preset —" },
    ...(presets.data ?? []).map((preset) => ({ value: preset.id, label: preset.name })),
  ];

  const load = () => {
    if (!selected) return;
    const metadata = selected.metadata;
    if (metadata.version !== PRESET_VERSION || metadata.tab !== tab) {
      throw new Error(`Training preset ${selected.name} is incompatible with the ${tab} tab`);
    }
    onLoad(structuredClone(metadata.graph as WorkflowGraph));
  };

  const save = async () => {
    const presetName = name.trim();
    if (!presetName) return;
    const created = await createPreset.mutateAsync({
      name: presetName,
      type_: type,
      metadata: { version: PRESET_VERSION, tab, graph: structuredClone(graph) },
    });
    setSelectedId(created.id);
    setName("");
  };

  const exportWorkflow = () => {
    const config = runtimeConfigForGraph(schema, graph, schema.runtime_config_defaults);
    const contents = JSON.stringify(workflowDefinition(graph, config), null, 2);
    const url = URL.createObjectURL(new Blob([contents], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "workflow.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const uploadWorkflow = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const definition = JSON.parse(await file.text()) as WorkflowDefinition;
      const uploadedGraph = graphFromDefinition(definition, tab);
      runtimeConfigForGraph(schema, uploadedGraph, definition.context.config);
      onLoad(uploadedGraph);
      setFileError("");
    } catch (error) {
      setFileError(error instanceof Error ? error.message : "Could not read workflow.json");
    }
  };

  return (
    <Card className="px-4 py-4">
      <div className="mb-3">
        <div className="text-sm font-bold text-txt">Training presets</div>
        <div className="mt-0.5 text-[11px] text-txt-mute">Save or restore every setting in this tab.</div>
      </div>
      <div className="flex gap-2">
        <Select value={selectedId} onChange={setSelectedId} options={options} variant="mini" className="min-w-0 flex-1" />
        <Button size="sm" disabled={!selected} onClick={load}>Load</Button>
      </div>
      <div className="mt-2 flex gap-2">
        <Input
          value={name}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void save();
          }}
          placeholder="Preset name"
          aria-label="Preset name"
        />
        <Button size="sm" variant="primary" disabled={!name.trim() || createPreset.isPending} onClick={() => void save()}>
          {createPreset.isPending ? "Saving…" : "Save"}
        </Button>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <Button size="sm" icon="download" onClick={exportWorkflow}>Export</Button>
        <Button size="sm" icon="upload" onClick={() => fileRef.current?.click()}>Upload</Button>
        <input ref={fileRef} type="file" accept=".json,application/json" className="hidden" onChange={(event) => void uploadWorkflow(event)} />
      </div>
      {presets.error ? <div className="mt-2 text-[11px] text-red-600">Could not load presets.</div> : null}
      {createPreset.error ? <div className="mt-2 text-[11px] text-red-600">Could not save preset.</div> : null}
      {fileError ? <div className="mt-2 text-[11px] text-red-600">{fileError}</div> : null}
    </Card>
  );
}

function graphFromDefinition(definition: WorkflowDefinition, tab: TrainTab): WorkflowGraph {
  if (!Array.isArray(definition.nodes) || !Array.isArray(definition.edges) || !definition.context) {
    throw new Error("The selected file is not a workflow.json file");
  }
  const expectedNodes = TRAINING_WORKFLOWS[tab].nodes;
  const nodesById = new Map(definition.nodes.map((node) => [node.id, node]));
  const incompatible = expectedNodes.find((expected) => nodesById.get(expected.id)?.type !== expected.type);
  if (incompatible) {
    throw new Error(`Workflow is incompatible with the ${tab} training tab`);
  }
  return {
    nodes: structuredClone(definition.nodes),
    edges: structuredClone(definition.edges),
    panels: structuredClone(definition.panels ?? []),
  };
}
