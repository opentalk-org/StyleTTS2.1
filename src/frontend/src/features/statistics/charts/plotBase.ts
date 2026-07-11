import type { Config, Layout } from "plotly.js";

import type { Tone } from "../logic";

const FONT = '"Outfit", system-ui, -apple-system, "Segoe UI", sans-serif';
const AXIS = "#9ca3af";
const GRID = "#eef0f3";
const LINE = "#e5e7eb";

export const PLOT_COLOR: Record<Tone, string> = {
  blue: "#3b82f6",
  emerald: "#10b981",
  amber: "#f59e0b",
  red: "#ef4444",
};

// Shared Plotly layout: transparent so it inherits the card background, app font,
// tight margins, and grid/axis colors matched to the design tokens in index.css.
export function baseLayout(overrides: Partial<Layout> = {}): Partial<Layout> {
  const { xaxis, yaxis, margin, ...rest } = overrides;
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: FONT, size: 11, color: AXIS },
    margin: { l: 46, r: 12, t: 6, b: 32, ...margin },
    hovermode: "closest",
    dragmode: "zoom",
    showlegend: false,
    bargap: 0.04,
    xaxis: { gridcolor: GRID, zerolinecolor: LINE, linecolor: LINE, automargin: true, ...xaxis },
    yaxis: { gridcolor: GRID, zerolinecolor: LINE, linecolor: LINE, automargin: true, ...yaxis },
    ...rest,
  };
}

// Wheel-zoom + pan + a clean modebar. Range change comes from drag-select (zoom),
// pan mode, and double-click to reset — all native Plotly interactions.
export const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  scrollZoom: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  toImageButtonOptions: { format: "png", scale: 2 },
};
