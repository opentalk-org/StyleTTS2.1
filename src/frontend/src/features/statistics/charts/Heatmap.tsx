import Plot from "react-plotly.js";

import type { BigramMatrix } from "../api";
import { baseLayout, PLOT_CONFIG } from "./plotBase";

const BLUE_SCALE: [number, string][] = [
  [0, "#f8fafc"],
  [0.15, "#dbeafe"],
  [0.4, "#93c5fd"],
  [0.7, "#3b82f6"],
  [1, "#1d4ed8"],
];

export function Heatmap({ data, unit, height = 600 }: { data: BigramMatrix; unit: string; height?: number }) {
  const { labels, matrix } = data;
  return (
    <Plot
      data={[
        {
          type: "heatmap",
          z: matrix,
          x: labels,
          y: labels,
          xgap: 1,
          ygap: 1,
          colorscale: BLUE_SCALE,
          hovertemplate: `%{y} → %{x}<br>%{z} occurrences<extra></extra>`,
          colorbar: { thickness: 10, outlinewidth: 0, tickfont: { size: 10 }, len: 0.9 },
        },
      ]}
      layout={baseLayout({
        margin: { l: 34, r: 12, t: 6, b: 34 },
        xaxis: { type: "category", side: "bottom", tickfont: { size: 10 }, title: { text: `second ${unit}`, font: { size: 11 } }, gridcolor: "transparent" },
        // Reverse so the first token reads top-to-bottom like the data, not bottom-up.
        yaxis: { type: "category", autorange: "reversed", tickfont: { size: 10 }, title: { text: `first ${unit}`, font: { size: 11 } }, gridcolor: "transparent" },
      })}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
