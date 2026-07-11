import Plot from "react-plotly.js";

import type { Tone } from "../logic";
import { baseLayout, PLOT_COLOR, PLOT_CONFIG } from "./plotBase";

// Interactive histogram: a real numeric x-axis built from the bin edges, so drag-to-zoom,
// pan, wheel-zoom, and double-click-reset all work and hover reports the exact bin range.
export function Histogram({
  edges,
  counts,
  tone = "blue",
  height = 132,
}: {
  edges: number[];
  counts: number[];
  tone?: Tone;
  height?: number;
}) {
  const centers = counts.map((_, i) => (edges[i]! + edges[i + 1]!) / 2);
  const widths = counts.map((_, i) => edges[i + 1]! - edges[i]!);
  const ranges = counts.map((_, i) => `${fmt(edges[i]!)} – ${fmt(edges[i + 1]!)}`);
  return (
    <Plot
      data={[
        {
          type: "bar",
          x: centers,
          y: counts,
          width: widths,
          marker: { color: PLOT_COLOR[tone], opacity: 0.9 },
          customdata: ranges,
          hovertemplate: "%{customdata}<br>%{y} files<extra></extra>",
        },
      ]}
      layout={baseLayout({ yaxis: { rangemode: "tozero" } })}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}

function fmt(value: number): string {
  const abs = Math.abs(value);
  if (abs !== 0 && abs < 1) return value.toFixed(2);
  if (abs < 100) return value.toFixed(1);
  return Math.round(value).toLocaleString();
}
