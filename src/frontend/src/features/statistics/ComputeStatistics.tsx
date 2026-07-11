import { useState } from "react";

import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";
import { useDatasetsQuery } from "../datasets/query";
import { useWorkflowSchemaQuery } from "../workflows/query";
import { useComputeStatisticsMutation } from "./query";

// Trigger the dataset-statistics workflow for a chosen dataset. The workflow builds and runs
// the AudioSource(dataset) → … → SaveStatisticsEntry pipeline and refreshes the entry list.
export function ComputeStatistics() {
  const datasets = useDatasetsQuery();
  const schemaQuery = useWorkflowSchemaQuery();
  const compute = useComputeStatisticsMutation();
  const [datasetId, setDatasetId] = useState("");
  const [name, setName] = useState("");

  const rows = datasets.data ?? [];
  const placeholder = { value: "", label: rows.length ? "Select dataset…" : "No datasets" };
  const options = [placeholder, ...rows.map((dataset) => ({ value: dataset.id, label: dataset.name }))];
  const selected = rows.find((dataset) => dataset.id === datasetId);

  const run = () => {
    if (!schemaQuery.data) {
      showToast("Workflow schema not loaded yet", undefined, "error");
      return;
    }
    if (!datasetId) {
      showToast("Choose a dataset first", undefined, "error");
      return;
    }
    const entryName = name.trim() || `${selected?.name ?? "Dataset"} statistics`;
    compute.mutate({ schema: schemaQuery.data, datasetId, name: entryName });
  };

  return (
    <div className="flex flex-wrap items-center gap-2.5">
      <div className="min-w-[200px]">
        <Select variant="mini" value={datasetId} onChange={setDatasetId} options={options} />
      </div>
      <div className="w-[200px]">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={selected ? `${selected.name} statistics` : "Entry name"} filled />
      </div>
      <Button variant="primary" size="sm" icon="bar-chart" disabled={compute.isPending} onClick={run}>
        {compute.isPending ? "Computing…" : "Compute statistics"}
      </Button>
    </div>
  );
}
