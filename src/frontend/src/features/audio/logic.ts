import type { Dataset } from "@/features/datasets/api";
import type { Option } from "@/shared/ui/Select";

export function datasetOptions(datasets: Dataset[]): Option[] {
  return [
    { value: "all", label: "All datasets" },
    { value: "unassigned", label: "Unassigned" },
    ...datasets.map((d) => ({ value: d.id, label: d.name })),
  ];
}

export function sortOptions(): Option[] {
  return [
    { value: "updated", label: "Sort: Recent" },
    { value: "name", label: "Sort: Name" },
    { value: "duration", label: "Sort: Duration" },
    { value: "segments", label: "Sort: Segments" },
  ];
}
