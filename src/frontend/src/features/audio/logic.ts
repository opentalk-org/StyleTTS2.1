import { AUDIO_COUNT } from "@/mock/data";
import type { Dataset } from "@/mock/types";
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
    { value: "speaker", label: "Sort: Speaker" },
    { value: "segments", label: "Sort: Segments" },
  ];
}

/**
 * Rows the virtualized table renders over.
 *
 * ponytail: real filtering is server-side, so the toolbar filters are controlled
 * UI state only and the table always virtualizes the full library count. Wire a
 * server count here when the backend can page filtered results.
 */
export function filteredAudioCount(): number {
  return AUDIO_COUNT;
}
