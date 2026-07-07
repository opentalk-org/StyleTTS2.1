import { useEffect } from "react";

import { Field } from "@/shared/ui/form/Field";
import { Select } from "@/shared/ui/Select";
import { useDatasetsQuery } from "../datasets/query";
import type { WorkflowGraph } from "../workflows/types";
import { testingNode, updateNodeParams } from "./logic";

export function DatasetField({
  graph,
  datasetNodeId,
  onChange,
}: {
  graph: WorkflowGraph;
  datasetNodeId: string;
  onChange: (graph: WorkflowGraph) => void;
}) {
  const datasets = useDatasetsQuery();
  const node = testingNode(graph, datasetNodeId);
  const current = String(node.params.dataset_id ?? "");
  const list = datasets.data ?? [];

  useEffect(() => {
    if (current || list.length === 0) return;
    const target = list.find((item) => item.name === "synthesis") ?? list[0];
    if (target) onChange(updateNodeParams(graph, datasetNodeId, { ...node.params, dataset_id: target.id }));
  }, [current, list.length]);

  return (
    <Field label="Save to dataset">
      <Select
        value={current}
        onChange={(dataset_id) => onChange(updateNodeParams(graph, datasetNodeId, { ...node.params, dataset_id }))}
        options={[
          { value: "", label: list.length ? "— select dataset —" : "No datasets" },
          ...list.map((item) => ({ value: item.id, label: item.name })),
        ]}
      />
    </Field>
  );
}
