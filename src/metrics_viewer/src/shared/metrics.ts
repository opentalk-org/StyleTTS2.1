import type { SearchOption } from "@/shared/ui";
import type { Run } from "@/shared/types";

const RUN_FIELDS: { id: string; label: string }[] = [
  { id: "name", label: "Run" },
  { id: "status", label: "Status" },
  { id: "startedAt", label: "Started" },
  { id: "duration", label: "Duration" },
];

const DEFAULT_RUN_FIELDS = RUN_FIELDS.map((field) => field.id);
const DEFAULT_METRIC_COLUMN_COUNT = 2;
const PREFERRED_METRIC_WORDS = ["loss", "val"];

/**
 * Deterministic default columns: the run fields plus the first two summary metrics
 * that look like a loss or validation metric, else the first two alphabetically.
 */
export function defaultRunColumns(runs: Run[]): string[] {
  const metrics = [...new Set(runs.flatMap((run) => Object.keys(run.summary)))].sort();
  const preferred = metrics.filter((name) =>
    PREFERRED_METRIC_WORDS.some((word) => name.toLowerCase().includes(word)),
  );
  const chosen = [...preferred, ...metrics.filter((name) => !preferred.includes(name))]
    .slice(0, DEFAULT_METRIC_COLUMN_COUNT)
    .map((name) => `metric:${name}`);
  return [...DEFAULT_RUN_FIELDS, ...chosen];
}

export function metricGroup(name: string): string {
  const separator = name.indexOf("/");
  return separator === -1 ? "other" : name.slice(0, separator);
}

export function metricLeaf(name: string): string {
  return name.split("/").at(-1) ?? name;
}

export function runColumnOptions(runs: Run[]): SearchOption[] {
  const params = [...new Set(runs.flatMap((run) => Object.keys(run.params)))].sort();
  const metrics = [...new Set(runs.flatMap((run) => Object.keys(run.summary)))].sort();
  return [
    ...RUN_FIELDS.map((field) => ({ value: field.id, label: field.label, group: "Run" })),
    ...params.map((name) => ({ value: `param:${name}`, label: name, group: "Parameters" })),
    ...metrics.map((name) => ({
      value: `metric:${name}`,
      label: metricLeaf(name),
      group: "Metrics",
      hint: metricGroup(name),
    })),
  ];
}

export function runColumnLabel(id: string): string {
  const field = RUN_FIELDS.find((candidate) => candidate.id === id);
  if (field !== undefined) return field.label;
  if (id.startsWith("param:")) return id.slice(6);
  if (id.startsWith("metric:")) return metricLeaf(id.slice(7));
  return id;
}

export function isNumericColumn(id: string): boolean {
  return id === "duration" || id.startsWith("metric:");
}
