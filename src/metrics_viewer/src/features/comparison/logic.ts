import type { Data, Layout } from "plotly.js";

import { axis, baseLayout, runColor } from "@/shared/chart";
import type { PlotRow, PlotSettings, Run } from "@/shared/types";

/** One run's line inside one plot, in the order the query returned it. */
export interface PlotSeries {
  runId: string;
  x: number[];
  y: number[];
}

export interface Plot {
  name: string;
  series: PlotSeries[];
  pointCount: number;
}

/**
 * Pivots the flat query result into charts. The query owns which plots exist and
 * what lands on each axis; this only groups by plot, then by run.
 */
export function groupPlots(rows: PlotRow[]): Plot[] {
  const plots = new Map<string, Map<string, PlotSeries>>();
  for (const row of rows) {
    const byRun = plots.get(row.plot) ?? new Map<string, PlotSeries>();
    const series = byRun.get(row.runId) ?? { runId: row.runId, x: [], y: [] };
    series.x.push(row.x);
    series.y.push(row.y);
    byRun.set(row.runId, series);
    plots.set(row.plot, byRun);
  }
  return [...plots.entries()].map(([name, byRun]) => {
    const series = [...byRun.values()];
    return {
      name,
      series,
      pointCount: series.reduce((total, item) => total + item.x.length, 0),
    };
  });
}

/** Plotly traces for one plot, ordered and coloured by the run table's order. */
export function buildTraces(
  plot: Plot,
  runs: Run[],
  hiddenRuns: Set<string>,
  settings: PlotSettings,
  runColors: Record<string, string>,
): Data[] {
  return runs.flatMap((run, position) => {
    if (hiddenRuns.has(run.id)) return [];
    const series = plot.series.find((item) => item.runId === run.id);
    if (series === undefined) return [];
    return seriesTraces(series, run.name, runColor(run.id, position, runColors), settings);
  });
}

function seriesTraces(
  series: PlotSeries,
  name: string,
  color: string,
  settings: PlotSettings,
): Data[] {
  const type = series.x.length > 350 ? "scattergl" : "scatter";
  const mode =
    settings.renderMode === "line"
      ? "lines"
      : settings.renderMode === "scatter"
        ? "markers"
        : "lines+markers";
  const base = { type, mode, x: series.x, hovertemplate: `%{y:.5g}<extra>${name}</extra>` } as const;
  if (settings.smoothing === "none") {
    return [
      {
        ...base,
        name,
        y: series.y,
        opacity: settings.smoothOpacity,
        line: { color, width: 2 },
        marker: { color, size: 4 },
      } as Data,
    ];
  }
  return [
    {
      ...base,
      name: `${name} raw`,
      y: series.y,
      opacity: settings.rawOpacity,
      line: { color, width: 1 },
      marker: { color, size: 3 },
      showlegend: false,
      hoverinfo: "skip",
    } as Data,
    {
      ...base,
      name,
      y: smoothValues(series.y, settings),
      opacity: settings.smoothOpacity,
      line: { color, width: 2 },
      marker: { color, size: 4 },
    } as Data,
  ];
}

/** The x position of a hovered point, used to draw the cursor line on every plot. */
export function cursorXAt(plot: Plot, runs: Run[], pointIndex: number | null): number | null {
  if (pointIndex === null) return null;
  const first = runs.map((run) => plot.series.find((item) => item.runId === run.id)).find(Boolean);
  return first?.x[pointIndex] ?? null;
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

export function plotLayout(
  settings: PlotSettings,
  cursorX: number | null,
  height?: number,
): Partial<Layout> {
  return {
    ...baseLayout(height),
    hovermode: "x unified",
    hoverdistance: -1,
    dragmode: "zoom",
    // Horizontal gridlines only: vertical divisions add noise to step-indexed curves.
    xaxis: axis({ showgrid: false, type: settings.xScale, fixedrange: false }),
    yaxis: axis({ showgrid: true, type: settings.yScale, fixedrange: false }),
    shapes:
      cursorX === null
        ? []
        : [
            {
              type: "line",
              x0: cursorX,
              x1: cursorX,
              xref: "x",
              y0: 0,
              y1: 1,
              yref: "paper",
              line: { color: "rgba(129,140,248,0.45)", width: 1, dash: "dot" },
              layer: "above",
            },
          ],
    uirevision: `${settings.xScale}-${settings.yScale}`,
  };
}
