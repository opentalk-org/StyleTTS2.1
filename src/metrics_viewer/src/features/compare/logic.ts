import type { Data, Layout } from "plotly.js";

import { durationMs, formatDuration, formatMetric, formatRelative } from "@/features/runs/logic";
import { axis, baseLayout, type ChartTheme } from "@/shared/chart";
import { runColumnLabel } from "@/shared/metrics";
import type { ComparePlotConfig, Run, Scalar } from "@/shared/types";
import { statusLabel } from "@/shared/ui";

export const RUN_AXIS = "run";
const MAX_TICK_CHARS = 18;
const DEFAULT_PARAM_COLUMNS = 6;
const DEFAULT_METRIC_COLUMNS = 4;
const PREFERRED_METRIC_WORDS = ["loss", "val", "acc"];

/** Raw cell value for sorting and plotting; undefined when the run has no such field. */
export function rawValue(run: Run, column: string): Scalar | undefined {
  if (column === "name") return run.name;
  if (column === "status") return run.status;
  if (column === "startedAt") return run.startedAt;
  if (column === "duration") return durationMs(run);
  if (column.startsWith("param:")) return numericString(run.params[column.slice(6)]);
  if (column.startsWith("metric:")) return run.summary[column.slice(7)];
  return undefined;
}

/** Training configs often store numbers as strings; treat those as numbers so they sort and plot. */
function numericString(value: Scalar | undefined): Scalar | undefined {
  if (typeof value !== "string" || value.trim() === "" || Number.isNaN(Number(value))) return value;
  return Number(value);
}

export function cellText(run: Run, column: string): string {
  const value = rawValue(run, column);
  if (value === undefined) return "—";
  if (column === "status") return statusLabel(run.status);
  if (column === "startedAt") return formatRelative(value as number);
  if (column === "duration") return formatDuration(value as number);
  if (column.startsWith("metric:")) return formatMetric(value as number);
  return String(value);
}

export function isNumeric(runs: Run[], column: string): boolean {
  const values = runs.map((run) => rawValue(run, column)).filter((value) => value !== undefined);
  return values.length > 0 && values.every((value) => typeof value === "number");
}

export function isIdentical(runs: Run[], column: string): boolean {
  return new Set(runs.map((run) => String(rawValue(run, column)))).size <= 1;
}

/** Parameters that differ between the runs, then the most interesting final metrics. */
export function defaultCompareColumns(runs: Run[]): string[] {
  const params = [...new Set(runs.flatMap((run) => Object.keys(run.params)))].sort();
  const differing = params.filter((name) => !isIdentical(runs, `param:${name}`)).slice(0, DEFAULT_PARAM_COLUMNS);
  const metrics = [...new Set(runs.flatMap((run) => Object.keys(run.summary)))].sort();
  const preferred = metrics.filter((name) => PREFERRED_METRIC_WORDS.some((word) => name.toLowerCase().includes(word)));
  const chosen = [...preferred, ...metrics.filter((name) => !preferred.includes(name))].slice(0, DEFAULT_METRIC_COLUMNS);
  return [...differing.map((name) => `param:${name}`), ...chosen.map((name) => `metric:${name}`)];
}

export function compareValue(a: Scalar | undefined, b: Scalar | undefined, direction: "asc" | "desc"): number {
  if (a === undefined || b === undefined) return Number(a === undefined) - Number(b === undefined);
  const compared = typeof a === "number" && typeof b === "number"
    ? a - b
    : String(a).localeCompare(String(b), undefined, { numeric: true });
  return (direction === "asc" ? 1 : -1) * compared;
}

export function axisLabel(column: string): string {
  return column === RUN_AXIS ? "Run" : runColumnLabel(column);
}

/**
 * One trace per run so every point carries the run colour. Numeric x gives a scatter,
 * categorical x (run name or a string parameter) gives bars.
 */
export function comparePlotTraces(
  runs: Run[],
  x: string,
  y: string,
  runColors: Record<string, string>,
  theme: ChartTheme,
): { traces: Data[]; categorical: boolean } {
  const categorical = x === RUN_AXIS || !isNumeric(runs, x);
  const traces = runs.flatMap((run) => {
    const yValue = rawValue(run, y);
    if (typeof yValue !== "number") return [];
    const xValue = x === RUN_AXIS ? run.name : rawValue(run, x);
    if (xValue === undefined) return [];
    const color = runColors[run.id];
    const hover = `${run.name}<br>${axisLabel(x)}: ${cellText(run, x === RUN_AXIS ? "name" : x)}<br>${axisLabel(y)}: ${formatMetric(yValue)}<extra></extra>`;
    const meta = { runId: run.id, opacity: 1 };
    if (categorical) {
      return [{ type: "bar", name: run.name, x: [String(xValue)], y: [yValue], marker: { color }, hovertemplate: hover, meta } as Data];
    }
    return [
      {
        type: "scatter",
        mode: "markers",
        name: run.name,
        x: [xValue as number],
        y: [yValue],
        marker: { color, size: 9, line: { color: theme.hoverBg, width: 1 } },
        hovertemplate: hover,
        meta,
      } as Data,
    ];
  });
  return { traces, categorical };
}

function shortLabel(text: string): string {
  return text.length > MAX_TICK_CHARS ? `${text.slice(0, MAX_TICK_CHARS)}…` : text;
}

export function comparePlotLayout(
  theme: ChartTheme,
  runs: Run[],
  config: ComparePlotConfig,
  x: string,
  y: string,
  categorical: boolean,
  height: number,
): Partial<Layout> {
  // Categories are keyed by the full run name so runs with similar names keep separate bars.
  const runTicks = x === RUN_AXIS ? { tickvals: runs.map((run) => run.name), ticktext: runs.map((run) => shortLabel(run.name)) } : {};
  return {
    ...baseLayout(theme, height),
    margin: { l: 56, r: 24, t: 16, b: 56 },
    barmode: "group",
    hovermode: "closest",
    xaxis: axis(theme, {
      title: { text: axisLabel(x), font: { size: 11 } },
      showgrid: false,
      type: categorical ? "category" : config.logX ? "log" : "linear",
      automargin: true,
      tickangle: 0,
      ...(categorical ? { nticks: 0, minor: {} } : {}),
      ...runTicks,
    }),
    yaxis: axis(theme, { title: { text: axisLabel(y), font: { size: 11 } }, showgrid: true, automargin: true, type: config.logY ? "log" : "linear" }),
  };
}
