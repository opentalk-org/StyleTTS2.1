import type { SearchOption } from "@/shared/ui";
import type { Run } from "@/shared/types";

const RUN_FIELDS: { id: string; label: string }[] = [
  { id: "name", label: "Run" },
  { id: "status", label: "Status" },
  { id: "startedAt", label: "Started" },
  { id: "duration", label: "Duration" },
];


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
