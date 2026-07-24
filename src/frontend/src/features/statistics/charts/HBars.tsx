import Plot from "react-plotly.js";

import type { HBarItem, Tone } from "../logic";
import { baseLayout, PLOT_COLOR, PLOT_CONFIG } from "./plotBase";

export function HBars({ items, tone = "emerald" }: { items: HBarItem[]; tone?: Tone }) {
  const height = Math.max(120, items.length * 24 + 34);
  return (
    <Plot
      data={[
        {
          type: "bar",
          orientation: "h",
          x: items.map((it) => it.value),
          y: items.map((it) => it.label),
          marker: { color: PLOT_COLOR[tone], opacity: 0.9 },
          customdata: items.map((it) => it.display),
          hovertemplate: "%{y}<br>%{customdata}<extra></extra>",
        },
      ]}
      layout={baseLayout({
        margin: { l: 116, r: 16, t: 6, b: 28 },
        yaxis: { autorange: "reversed", type: "category" },
        xaxis: { rangemode: "tozero" },
      })}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
