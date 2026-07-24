import Plot from "react-plotly.js";

import type { ScatterPoint } from "../api";
import type { Tone } from "../logic";
import { baseLayout, PLOT_COLOR, PLOT_CONFIG } from "./plotBase";

export function Scatter({
  points,
  xLabel,
  yLabel,
  tone = "blue",
  height = 260,
}: {
  points: ScatterPoint[];
  xLabel: string;
  yLabel: string;
  tone?: Tone;
  height?: number;
}) {
  return (
    <Plot
      data={[
        {
          type: "scatter",
          mode: "markers",
          x: points.map((p) => p[0]),
          y: points.map((p) => p[1]),
          marker: { color: PLOT_COLOR[tone], size: 5, opacity: 0.55 },
          hovertemplate: `${xLabel}: %{x:.2f}<br>${yLabel}: %{y:.2f}<extra></extra>`,
        },
      ]}
      layout={baseLayout({
        margin: { l: 52, r: 12, t: 6, b: 42 },
        xaxis: { title: { text: xLabel, font: { size: 11 } }, rangemode: "tozero" },
        yaxis: { title: { text: yLabel, font: { size: 11 } }, rangemode: "tozero" },
      })}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
