import type { Run, RunStatus, Scalar } from "@/shared/types";

export type SortDirection = "asc" | "desc";

export interface RunSort {
  column: string;
  direction: SortDirection;
}

export const STATUS_ORDER: RunStatus[] = ["running", "queued", "succeeded", "failed", "cancelled"];

export interface RunFilter {
  query: string;
  statuses: RunStatus[];
}

export function filterRuns(runs: Run[], filter: RunFilter, sort: RunSort | null, starred: string[]): Run[] {
  const normalized = filter.query.trim().toLowerCase();
  const matches = runs.filter(
    (run) =>
      (filter.statuses.length === 0 || filter.statuses.includes(run.status)) &&
      (normalized.length === 0 ||
        `${run.name} ${run.status} ${Object.values(run.params).join(" ")}`.toLowerCase().includes(normalized)),
  );
  const ordered =
    sort === null
      ? matches
      : [...matches].sort(
          (a, b) => compareSortValues(sortValue(a, sort.column), sortValue(b, sort.column), sort.direction),
        );
  return [...ordered.filter((run) => starred.includes(run.id)), ...ordered.filter((run) => !starred.includes(run.id))];
}

export function durationMs(run: Run): number {
  if (run.endedAt === 0) return Date.now() - run.startedAt;
  return run.endedAt - run.startedAt;
}

export function formatDuration(ms: number): string {
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return "<1m";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

export function formatRelative(value: number): string {
  if (value === 0) return "never";
  const minutes = Math.round((Date.now() - value) / 60000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} h`;
  return `${Math.round(hours / 24)} d`;
}

export function formatAbsolute(value: number): string {
  return new Date(value).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatMetric(value: number): string {
  if (Number.isNaN(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e5 || magnitude < 1e-3)) return value.toExponential(3);
  return value.toFixed(4);
}

export function sortValue(run: Run, column: string): Scalar | undefined {
  if (column === "name") return run.name;
  if (column === "status") return STATUS_ORDER.indexOf(run.status);
  if (column === "startedAt") return run.startedAt;
  if (column === "duration") return durationMs(run);
  if (column.startsWith("param:")) return run.params[column.slice(6)];
  if (column.startsWith("metric:")) return run.summary[column.slice(7)];
  return undefined;
}

function compareSortValues(a: Scalar | undefined, b: Scalar | undefined, direction: SortDirection): number {
  const aMissing = a === undefined || (typeof a === "number" && Number.isNaN(a));
  const bMissing = b === undefined || (typeof b === "number" && Number.isNaN(b));
  if (aMissing || bMissing) return Number(aMissing) - Number(bMissing);
  return (direction === "asc" ? 1 : -1) * compareValues(a, b);
}

function compareValues(a: Scalar, b: Scalar): number {
  if (typeof a === "number" && typeof b === "number") {
    return a - b;
  }
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function cellText(run: Run, column: string): string {
  if (column === "name") return run.name;
  if (column === "startedAt") return formatRelative(run.startedAt);
  if (column === "duration") return formatDuration(durationMs(run));
  if (column.startsWith("param:")) return String(run.params[column.slice(6)] ?? "—");
  if (column.startsWith("metric:")) {
    const value = run.summary[column.slice(7)];
    return value === undefined ? "—" : formatMetric(value);
  }
  return "—";
}

export function columnWidth(column: string): string {
  if (column === "name") return "minmax(180px,2fr)";
  if (column === "status") return "104px";
  if (column === "startedAt") return "72px";
  if (column === "duration") return "84px";
  return "minmax(96px,1fr)";
}

export function columnMinWidth(column: string): number {
  if (column === "name") return 180;
  if (column === "status") return 104;
  if (column === "startedAt") return 72;
  if (column === "duration") return 84;
  return 96;
}
