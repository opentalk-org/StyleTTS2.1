import type { Layout } from "plotly.js";

import type { Run } from "@/shared/types";

const SERIES_COUNT = 12;

export interface ChartTheme {
  series: string[];
  grid: string;
  axisText: string;
  cursor: string;
  hoverBg: string;
  hoverBorder: string;
  hoverText: string;
  rawOpacity: number;
  sans: string;
  mono: string;
}

/** Reads the chart tokens off the document so Plotly follows the active theme. */
export function readChartTheme(): ChartTheme {
  const style = getComputedStyle(document.documentElement);
  const token = (name: string) => style.getPropertyValue(name).trim();
  return {
    series: Array.from({ length: SERIES_COUNT }, (_, index) => token(`--series-${index + 1}`)),
    grid: token("--chart-grid"),
    axisText: token("--fg-muted"),
    cursor: token("--fg-secondary"),
    hoverBg: token("--raised"),
    hoverBorder: token("--strong"),
    hoverText: token("--fg"),
    rawOpacity: Number(token("--chart-raw-opacity")),
    sans: token("--font-sans"),
    mono: token("--font-mono"),
  };
}

/**
 * Colour by position in the project's chronological run order so a run keeps its
 * colour while other runs are selected or deselected. `runs` is the full project list
 * ordered newest first, as returned by the runs query.
 */
export function assignRunColors(
  runs: Run[],
  palette: string[],
  overrides: Record<string, string>,
): Record<string, string> {
  const colors: Record<string, string> = {};
  const oldestFirst = [...runs].sort((a, b) => a.startedAt - b.startedAt);
  oldestFirst.forEach((run, index) => {
    colors[run.id] = overrides[run.id] ?? palette[index % palette.length];
  });
  return colors;
}

export function baseLayout(theme: ChartTheme, height?: number): Partial<Layout> {
  return {
    autosize: true,
    ...(height === undefined ? {} : { height }),
    margin: { l: 48, r: 12, t: 8, b: 32 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: theme.axisText, family: theme.sans, size: 11 },
    showlegend: false,
    hoverlabel: {
      bgcolor: theme.hoverBg,
      bordercolor: theme.hoverBorder,
      font: { color: theme.hoverText, family: theme.mono, size: 11 },
      align: "left",
      namelength: -1,
    },
  };
}

/** Half-strength version of a #rrggbb colour for minor gridlines. */
function faint(hex: string): string {
  const value = Number.parseInt(hex.slice(1), 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, 0.45)`;
}

export function axis(
  theme: ChartTheme,
  overrides: Partial<Layout["xaxis"]> = {},
): Partial<Layout["xaxis"]> {
  return {
    gridcolor: theme.grid,
    gridwidth: 1,
    zeroline: false,
    showline: false,
    showspikes: false,
    nticks: 10,
    ticks: "outside",
    ticklen: 4,
    tickcolor: theme.grid,
    tickfont: { color: theme.axisText, family: theme.mono, size: 11 },
    minor: { nticks: 5, ticks: "outside", ticklen: 2, tickcolor: theme.grid, showgrid: true, gridcolor: faint(theme.grid), gridwidth: 1 },
    ...overrides,
  };
}
