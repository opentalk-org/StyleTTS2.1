import Plotly from "plotly.js-basic-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

/** One Plotly component instance shared by every chart; import this module lazily. */
export const Plot = createPlotlyComponent(Plotly);

export { Plotly };

/** Charts inside the scrolling grid: drag to zoom, the wheel scrolls the page. */
export const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  displayModeBar: false,
  scrollZoom: false,
  doubleClick: "reset",
} as const;

/** The expanded chart is not inside a scrolling page, so the wheel can zoom. */
export const EXPANDED_PLOT_CONFIG = { ...PLOT_CONFIG, scrollZoom: true } as const;
