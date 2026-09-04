import type { SearchOption } from "@/shared/ui";
import type { Run } from "@/shared/types";

const RUN_FIELDS: { id: string; label: string }[] = [
  { id: "name", label: "Run" },
  { id: "status", label: "Status" },
  { id: "startedAt", label: "Started" },
  { id: "duration", label: "Duration" },
];

const DEFAULT_RUN_FIELDS = RUN_FIELDS.map((field) => field.id);
const DEFAULT_DYNAMIC_COLUMN_COUNT = 3;


export function defaultRunColumns(runs: Run[]): string[] {
  const available = runColumnOptions(runs)
    .map((option) => option.value)
    .filter((column) => !DEFAULT_RUN_FIELDS.includes(column));
  for (let index = available.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [available[index], available[swapIndex]] = [available[swapIndex], available[index]];
  }
  return [...DEFAULT_RUN_FIELDS, ...available.slice(0, DEFAULT_DYNAMIC_COLUMN_COUNT)];
}


export function metricGroup(name: string): string {
  const separator = name.indexOf("/");
  return separator === -1 ? "other" : name.slice(0, separator);
}


export function runColumnOptions(runs: Run[]): SearchOption[] {
  const params = [...new Set(runs.flatMap((run) => Object.keys(run.params)))].sort();
  const metrics = [...new Set(runs.flatMap((run) => Object.keys(run.summary)))].sort();
  return [
    ...RUN_FIELDS.map((field) => ({ value: field.id, label: field.label, group: "Run" })),
    ...params.map((name) => ({ value: `param:${name}`, label: name, group: "Parameters" })),
    ...metrics.map((name) => ({
      value: `metric:${name}`,
      label: name.split("/").at(-1) ?? name,
      group: "Metrics",
      hint: metricGroup(name),
    })),
  ];
}

export function runColumnLabel(id: string): string {
  const field = RUN_FIELDS.find((candidate) => candidate.id === id);
  if (field !== undefined) return field.label;
  if (id.startsWith("param:")) return id.slice(6);
  if (id.startsWith("metric:")) return id.slice(7).split("/").at(-1) ?? id;
  return id;
}
