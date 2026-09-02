import type { SearchOption } from "@/shared/ui";


export const METRIC_NAMES = [
  "train/total",
  "train/mel",
  "train/wavlm",
  "performance/steps_per_second",
  "performance/eta_hours",
] as const;


export const PARAM_NAMES = [
  "decoder",
  "learning_rate",
  "batch_seconds",
  "seed",
  "mixed_precision",
] as const;

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


export function runColumnOptions(): SearchOption[] {
  return [
    ...RUN_FIELDS.map((field) => ({ value: field.id, label: field.label, group: "Run" })),
    ...PARAM_NAMES.map((name) => ({ value: `param:${name}`, label: name, group: "Parameters" })),
    ...METRIC_NAMES.map((name) => ({
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
