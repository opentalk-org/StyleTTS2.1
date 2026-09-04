import type { Data, Layout } from "plotly.js";

import { axis, baseLayout, type ChartTheme } from "@/shared/chart";
import type { Artifact, GlobalPlotSettings, PlotQueryResult, PlotSettings, Run, XAxis } from "@/shared/types";

export interface PlotSeries {
  runId: string;
  step: number[];
  wall: number[] | null;
  rel: number[] | null;
  y: number[];
}

/** Settings with "inherit" resolved: what a chart actually draws. */
export interface EffectiveSettings extends PlotSettings {
  axis: XAxis;
}

export function seriesX(series: PlotSeries, axis: XAxis): number[] {
  if (axis === "wall" && series.wall !== null) return series.wall;
  if (axis === "relative" && series.rel !== null) return series.rel;
  return series.step;
}

export interface Plot {
  name: string;
  series: PlotSeries[];
  pointCount: number;
}

export function groupPlots(result: PlotQueryResult | null): Plot[] {
  if (result === null) return [];
  const plots = new Map<string, Map<string, PlotSeries>>();
  for (let index = 0; index < result.x.length; index += 1) {
    const name = result.plot[index];
    const runId = result.runId[index];
    const byRun = plots.get(name) ?? new Map<string, PlotSeries>();
    const series =
      byRun.get(runId) ??
      { runId, step: [], wall: result.wall === null ? null : [], rel: result.rel === null ? null : [], y: [] };
    series.step.push(result.x[index]);
    series.y.push(result.y[index]);
    if (result.wall !== null) series.wall?.push(result.wall[index]);
    if (result.rel !== null) series.rel?.push(result.rel[index]);
    byRun.set(runId, series);
    plots.set(name, byRun);
  }
  return [...plots.entries()].map(([name, byRun]) => {
    const series = [...byRun.values()];
    return { name, series, pointCount: series.reduce((total, item) => total + item.step.length, 0) };
  });
}

export function namespaceOf(name: string): string {
  const separator = name.indexOf("/");
  return separator === -1 ? "other" : name.slice(0, separator);
}

export interface Section<T> {
  name: string;
  items: T[];
  pinned: boolean;
}

/** Groups items by their `name` prefix; pinned sections come first, then alphabetical. */
export function sectionize<T extends { name: string }>(items: T[], pinned: string[]): Section<T>[] {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const key = namespaceOf(item.name);
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }
  return [...groups.entries()]
    .map(([name, group]) => ({ name, items: group, pinned: pinned.includes(name) }))
    .sort((a, b) => Number(b.pinned) - Number(a.pinned) || a.name.localeCompare(b.name));
}

export function artifactNames(artifacts: Artifact[]): { name: string }[] {
  return [...new Set(artifacts.map((artifact) => artifact.name))].sort().map((name) => ({ name }));
}

export function resolveSettings(settings: PlotSettings, global: GlobalPlotSettings): EffectiveSettings {
  const axis = settings.xAxis === "inherit" ? global.xAxis : settings.xAxis;
  if (settings.smoothing !== "inherit") return { ...settings, axis };
  return global.smoothing > 0
    ? { ...settings, axis, smoothing: "ema", smoothingValue: global.smoothing }
    : { ...settings, axis, smoothing: "none" };
}

export function buildTraces(
  plot: Plot,
  runs: Run[],
  settings: EffectiveSettings,
  runColors: Record<string, string>,
  theme: ChartTheme,
): Data[] {
  return runs.flatMap((run) => {
    const series = plot.series.find((item) => item.runId === run.id);
    if (series === undefined) return [];
    return seriesTraces(series, run.name, runColors[run.id], settings, theme.rawOpacity);
  });
}

function seriesTraces(
  series: PlotSeries,
  name: string,
  color: string,
  settings: EffectiveSettings,
  rawOpacity: number,
): Data[] {
  const x = seriesX(series, settings.axis);
  const type = x.length > 350 ? "scattergl" : "scatter";
  const mode =
    settings.renderMode === "line" ? "lines" : settings.renderMode === "scatter" ? "markers" : "lines+markers";
  // Hover labels are drawn by the chart card itself; Plotly only reports the hovered x.
  const base = { type, mode, x, hoverinfo: "none" } as const;
  if (settings.smoothing === "none") {
    return [{ ...base, name, y: series.y, meta: { runId: series.runId, opacity: 1 }, line: { color, width: 1.5 }, marker: { color, size: 4 } } as Data];
  }
  return [
    {
      ...base,
      name: `${name} raw`,
      y: series.y,
      meta: { runId: series.runId, opacity: rawOpacity },
      opacity: rawOpacity,
      line: { color, width: 1 },
      marker: { color, size: 3 },
      showlegend: false,
      hoverinfo: "skip",
    } as Data,
    { ...base, name, y: smoothValues(series.y, settings), meta: { runId: series.runId, opacity: 1 }, line: { color, width: 2 }, marker: { color, size: 4 } } as Data,
  ];
}

/** The y of each series at the sample nearest to `x`, or the last sample when x is null. */
export function valuesAt(plot: Plot, axis: XAxis, x: number | null): Map<string, number> {
  const values = new Map<string, number>();
  for (const series of plot.series) {
    if (series.y.length === 0) continue;
    values.set(series.runId, series.y[x === null ? series.y.length - 1 : nearestIndex(seriesX(series, axis), x)]);
  }
  return values;
}

function nearestIndex(sorted: number[], target: number): number {
  let low = 0;
  let high = sorted.length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sorted[mid] < target) low = mid + 1;
    else high = mid;
  }
  if (low > 0 && Math.abs(sorted[low - 1] - target) <= Math.abs(sorted[low] - target)) return low - 1;
  return low;
}

function smoothValues(values: number[], settings: PlotSettings): number[] {
  if (settings.smoothing === "ema") return exponentialMean(values, settings.smoothingValue);
  if (settings.smoothing === "mean") return rollingMean(values, Math.round(settings.smoothingValue));
  return values;
}

function exponentialMean(values: number[], weight: number): number[] {
  if (values.length === 0) return values;
  const result = [values[0]];
  for (let index = 1; index < values.length; index += 1) {
    result.push(result[index - 1] * weight + values[index] * (1 - weight));
  }
  return result;
}

function rollingMean(values: number[], window: number): number[] {
  let sum = 0;
  return values.map((value, index) => {
    sum += value;
    if (index >= window) sum -= values[index - window];
    return sum / Math.min(index + 1, window);
  });
}

export function plotLayout(settings: EffectiveSettings, theme: ChartTheme, height?: number): Partial<Layout> {
  return {
    ...baseLayout(theme, height),
    hovermode: "x",
    hoverdistance: -1,
    dragmode: "zoom",
    xaxis: axis(theme, { showgrid: false, type: settings.axis === "wall" ? "date" : settings.xScale, fixedrange: false }),
    yaxis: axis(theme, { showgrid: true, type: settings.yScale, fixedrange: false }),
    uirevision: `${settings.xScale}-${settings.yScale}-${settings.axis}`,
  };
}

export interface SeriesStats {
  last: number;
  min: number;
  minX: number;
  max: number;
  maxX: number;
}

export function seriesStats(series: PlotSeries, axis: XAxis): SeriesStats {
  const x = seriesX(series, axis);
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let minX = 0;
  let maxX = 0;
  series.y.forEach((value, index) => {
    if (value < min) {
      min = value;
      minX = x[index];
    }
    if (value > max) {
      max = value;
      maxX = x[index];
    }
  });
  return { last: series.y[series.y.length - 1], min, minX, max, maxX };
}

export function formatX(axis: XAxis, x: number): string {
  if (axis === "wall") return new Date(x).toLocaleTimeString();
  if (axis === "relative") return `${x.toFixed(x >= 100 ? 0 : 1)} s`;
  return Number.isInteger(x) ? x.toLocaleString() : x.toPrecision(4);
}
