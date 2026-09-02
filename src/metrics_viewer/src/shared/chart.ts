import type { Layout } from "plotly.js";


export const SERIES_COLORS = ["#818cf8", "#6366f1", "#a5b4fc", "#7dd3fc", "#c4b5fd", "#94a3b8"] as const;

export function seriesColor(index: number): string {
  return SERIES_COLORS[index % SERIES_COLORS.length];
}





export const RUN_COLOR_PALETTE = [
  "#818cf8",
  "#6366f1",
  "#a78bfa",
  "#38bdf8",
  "#22d3ee",
  "#2dd4bf",
  "#4ade80",
  "#fbbf24",
  "#fb7185",
  "#94a3b8",
] as const;


export function runColor(runId: string, index: number, overrides: Record<string, string>): string {
  return overrides[runId] ?? seriesColor(index);
}

const SANS = "Inter, Geist, system-ui, sans-serif";
const MONO = "Geist Mono, JetBrains Mono, SFMono-Regular, ui-monospace, monospace";

const GRID = "rgba(255,255,255,0.05)";
const AXIS_TEXT = "#71717a";



export function baseLayout(height?: number): Partial<Layout> {
  return {
    autosize: true,
    ...(height === undefined ? {} : { height }),
    margin: { l: 54, r: 16, t: 12, b: 40 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: AXIS_TEXT, family: SANS, size: 11 },
    showlegend: false,
    hoverlabel: {
      bgcolor: "#0d0d10",
      bordercolor: "rgba(255,255,255,0.10)",
      font: { color: "#f4f4f5", family: MONO, size: 11 },
      align: "left",
      namelength: -1,
    },
  };
}


export function axis(overrides: Partial<Layout["xaxis"]> = {}): Partial<Layout["xaxis"]> {
  return {
    gridcolor: GRID,
    gridwidth: 1,
    zeroline: false,
    showline: false,
    showspikes: false,
    ticks: "",
    tickfont: { color: AXIS_TEXT, family: MONO, size: 10 },
    ...overrides,
  };
}
