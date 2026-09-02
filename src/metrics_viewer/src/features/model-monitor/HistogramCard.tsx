import { Pause, Play, Settings2, SkipBack, SkipForward } from "lucide-react";
import Plotly from "plotly.js-basic-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { useEffect, useMemo, useState } from "react";

import { IconButton } from "@/shared/ui";
import { useArrayMetric } from "./query";

const Plot = createPlotlyComponent(Plotly);

export function HistogramCard({ runId, name, running }: {
  runId: string;
  name: string;
  running: boolean;
}) {
  const query = useArrayMetric(runId, name, running);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [bins, setBins] = useState(64);
  const series = query.data;
  const last = Math.max((series?.steps.length ?? 1) - 1, 0);

  useEffect(() => {
    if (!playing || index >= last) return;
    const timer = window.setInterval(() => setIndex((current) => Math.min(current + 1, last)), 200);
    return () => window.clearInterval(timer);
  }, [playing, index, last]);

  useEffect(() => {
    if (playing && index === last) setPlaying(false);
  }, [playing, index, last]);

  const histogram = useMemo(() => metricHistogram(series?.values[index], bins), [series, index, bins]);
  const step = series?.steps[index] ?? 0;

  return (
    <section className="flex min-h-[385px] min-w-0 flex-col rounded-md bg-inset p-2">
      <h3 className="m-0 truncate text-center font-mono text-sm font-semibold text-fg">{name}</h3>
      <Plot
        data={[{
          type: "bar",
          x: histogram.x,
          y: histogram.y,
          marker: { color: "#2563eb" },
          hovertemplate: "%{x:.5g}<br>%{y}<extra></extra>",
        }]}
        layout={{
          height: 315,
          margin: { l: 38, r: 4, t: 8, b: 26 },
          paper_bgcolor: "#090a0d",
          plot_bgcolor: "#07080a",
          font: { color: "#737783", size: 9 },
          bargap: 0.04,
          xaxis: { gridcolor: "rgba(255,255,255,.05)", zerolinecolor: "rgba(255,255,255,.12)" },
          yaxis: { gridcolor: "rgba(255,255,255,.05)", zeroline: false },
          showlegend: false,
        }}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
        className="w-full"
      />
      <div className="mt-auto text-center font-mono text-xs text-fg-secondary">
        Step {step} of {series?.steps[last] ?? 0}
      </div>
      <div className="mt-1 flex items-center gap-1">
        <IconButton label="Previous step" size="sm" onClick={() => setIndex(Math.max(index - 1, 0))}>
          <SkipBack size={14} />
        </IconButton>
        <IconButton label={playing ? "Pause" : "Play"} size="sm" onClick={() => setPlaying(!playing)}>
          {playing ? <Pause size={14} /> : <Play size={14} />}
        </IconButton>
        <IconButton label="Next step" size="sm" onClick={() => setIndex(Math.min(index + 1, last))}>
          <SkipForward size={14} />
        </IconButton>
        <input
          aria-label={`${name} step`}
          className="min-w-0 flex-1"
          type="range"
          min={0}
          max={last}
          value={Math.min(index, last)}
          onChange={(event) => setIndex(Number(event.target.value))}
        />
        <IconButton
          label={`${bins} bins`}
          size="sm"
          onClick={() => setBins(bins === 64 ? 16 : bins * 2)}
        >
          <Settings2 size={14} />
        </IconButton>
      </div>
    </section>
  );
}

function metricHistogram(value: number[] | undefined, bins: number) {
  if (value === undefined || value.length < 3) return { x: [], y: [] };
  const [lower, upper, ...source] = value;
  const stride = 64 / bins;
  const y = Array.from({ length: bins }, (_, index) =>
    source.slice(index * stride, (index + 1) * stride).reduce((sum, count) => sum + count, 0),
  );
  const width = upper === lower ? 1 : (upper - lower) / bins;
  const x = Array.from({ length: bins }, (_, index) => lower + (index + 0.5) * width);
  return { x, y };
}
