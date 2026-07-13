import Plot from "react-plotly.js";

import type { Tone } from "../logic";
import { histogramBars } from "./histogramGeometry";
import { baseLayout, PLOT_COLOR, PLOT_CONFIG } from "./plotBase";

// Interactive histogram: a real numeric x-axis built from the bin edges, so drag-to-zoom,
// pan, wheel-zoom, and double-click-reset all work and hover reports the exact bin range.
export function Histogram({
  edges,
  counts,
  tone = "blue",
  height = 132,
  countLabel = "files",
  underflow = 0,
  overflow = 0,
}: {
  edges: number[];
  counts: number[];
  tone?: Tone;
  height?: number;
  countLabel?: string;
  underflow?: number;
  overflow?: number;
}) {
  const bars = histogramBars(edges, counts, underflow, overflow);
  return (
    <Plot
      data={[
        {
          type: "bar",
          x: bars.centers,
          y: bars.counts,
          width: bars.widths,
          marker: { color: PLOT_COLOR[tone], opacity: 0.9 },
          customdata: bars.ranges,
          hovertemplate: `%{customdata}<br>%{y} ${countLabel}<extra></extra>`,
        },
      ]}
      layout={baseLayout({ yaxis: { rangemode: "tozero" } })}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
