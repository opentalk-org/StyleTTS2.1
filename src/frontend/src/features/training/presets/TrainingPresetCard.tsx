import { useState } from "react";

import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";

import type { WorkflowGraph } from "../../workflows/types";
import { useCreateTrainingConfigMutation, useTrainingConfigsQuery } from "../query";
import type { TrainTab } from "../store";

const PRESET_VERSION = 1;

export function TrainingPresetCard({
  tab,
  graph,
  onLoad,
}: {
  tab: TrainTab;
  graph: WorkflowGraph;
  onLoad: (graph: WorkflowGraph) => void;
}) {
  const type = `training_preset_${tab}`;
  const presets = useTrainingConfigsQuery(type);
  const createPreset = useCreateTrainingConfigMutation(type);
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("");
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
      {presets.error ? <div className="mt-2 text-[11px] text-red-600">Could not load presets.</div> : null}
      {createPreset.error ? <div className="mt-2 text-[11px] text-red-600">Could not save preset.</div> : null}
    </Card>
  );
}
