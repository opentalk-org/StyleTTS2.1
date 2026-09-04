import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { baseLayout, axis, type ChartTheme } from "@/shared/chart";
import { Plot } from "@/shared/plot";
import { Card, IconButton, Range, SegmentedControl, Skeleton } from "@/shared/ui";

import { useArrayMetric } from "./query";

const BIN_OPTIONS = [
  { value: 16, label: "16" },
  { value: 32, label: "32" },
  { value: 64, label: "64" },
];

export function HistogramCard({ runId, name, running, chart }: { runId: string; name: string; running: boolean; chart: ChartTheme }) {
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
  const xRange = useMemo(() => histogramRange(series?.values), [series]);
  const step = series?.steps[index] ?? 0;
  const kind = name.startsWith("grad/") ? "Gradient" : "Parameter";

  return (
    <Card>
      <div className="flex h-card-head items-center justify-between gap-2 border-b border-line px-3">
        <span className="min-w-0 truncate text-[13px] font-medium text-fg" title={name}>
          <span className="text-fg-muted">{kind} · </span>
          {name.split(".").at(-1)}
        </span>
        <SegmentedControl label="Bins" options={BIN_OPTIONS} value={bins} onValue={setBins} className="h-6" />
      </div>
      {series === undefined ? (
        <Skeleton className="m-2 h-44" />
      ) : (
        <div className="h-[180px] bg-plot">
          <Plot
            data={[{ type: "bar", x: histogram.x, y: histogram.y, marker: { color: chart.series[0] }, hovertemplate: "%{x:.5g}<br>%{y}<extra></extra>" }]}
            layout={{
              ...baseLayout(chart, 180),
              margin: { l: 40, r: 8, t: 8, b: 28 },
              bargap: 0.05,
              xaxis: axis(chart, { showgrid: false, range: xRange }),
              yaxis: axis(chart, { showgrid: true }),
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      )}
      <div className="flex items-center gap-1 border-t border-line px-2 py-1">
        <IconButton label="Previous step" size="sm" onClick={() => setIndex(Math.max(index - 1, 0))}>
          <SkipBack size={13} />
        </IconButton>
        <IconButton label={playing ? "Pause" : "Play"} size="sm" onClick={() => setPlaying(!playing)}>
          {playing ? <Pause size={13} /> : <Play size={13} />}
        </IconButton>
        <IconButton label="Next step" size="sm" onClick={() => setIndex(Math.min(index + 1, last))}>
          <SkipForward size={13} />
        </IconButton>
        <Range aria-label={`${name} step`} min={0} max={last} step={1} value={Math.min(index, last)} onValue={setIndex} className="min-w-0 flex-1" />
        <span className="w-24 shrink-0 text-right font-mono text-xs tabular-nums text-fg-muted">
          step <span className="text-fg">{step.toLocaleString()}</span>
        </span>
      </div>
    </Card>
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

function histogramRange(values: number[][] | undefined): [number, number] | undefined {
  if (values === undefined || values.length === 0) return undefined;
  const lower = Math.min(...values.map((value) => value[0]));
  const upper = Math.max(...values.map((value) => value[1]));
  if (lower !== upper) return [lower, upper];
  const padding = Math.max(Math.abs(lower) * 0.05, 1e-6);
  return [lower - padding, upper + padding];
}
