import { RotateCcw, Settings2, X } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import type { ChartTheme } from "@/shared/chart";
import { useCursorStore, useRunHighlight } from "@/shared/cursor";
import { runColumnOptions } from "@/shared/metrics";
import { Plot, PLOT_CONFIG, Plotly } from "@/shared/plot";
import type { ComparePlotConfig, Run } from "@/shared/types";
import { Card, EmptyState, Field, IconButton, Popover, SearchSelect, SegmentedControl } from "@/shared/ui";

import { axisLabel, comparePlotLayout, comparePlotTraces, isNumeric, RUN_AXIS } from "./logic";

const PLOT_HEIGHT = 320;

const SCALE_OPTIONS = [
  { value: "linear" as const, label: "Linear" },
  { value: "log" as const, label: "Log" },
];

interface ComparePlotProps {
  runs: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
  config: ComparePlotConfig;
  onConfig: (patch: Partial<ComparePlotConfig>) => void;
  onRemove: () => void;
}

/**
 * A compare plot card, styled like a metric chart card: title, hover actions, settings
 * popover. Hovering a point highlights that run in the run list and dims the rest.
 */
export function ComparePlot({ runs, runColors, chart, config, onConfig, onRemove }: ComparePlotProps) {
  const graphRef = useRef<HTMLElement | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const setCursor = useCursorStore((state) => state.set);
  const clearCursor = useCursorStore((state) => state.clear);
  useRunHighlight(graphRef);

  const fields = useMemo(() => runColumnOptions(runs).filter((option) => option.value !== "name"), [runs]);
  const xOptions = useMemo(() => [{ value: RUN_AXIS, label: "Run", group: "Rows" }, ...fields], [fields]);
  const yOptions = useMemo(() => fields.filter((option) => isNumeric(runs, option.value)), [fields, runs]);
  const known = useMemo(() => new Set(fields.map((option) => option.value)), [fields]);
  const y = config.y !== null && known.has(config.y) ? config.y : null;
  const x = config.x === RUN_AXIS || known.has(config.x) ? config.x : RUN_AXIS;
  const built = useMemo(
    () => (y === null ? null : comparePlotTraces(runs, x, y, runColors, chart)),
    [runs, x, y, runColors, chart],
  );
  const layout = useMemo(
    () => (y === null || built === null ? null : comparePlotLayout(chart, runs, config, x, y, built.categorical, PLOT_HEIGHT)),
    [chart, runs, config, x, y, built],
  );
  const numericX = built?.categorical === false;

  function resetZoom() {
    if (graphRef.current !== null) void Plotly.relayout(graphRef.current, { "xaxis.autorange": true, "yaxis.autorange": true });
  }

  const settings = (
    <div className="flex flex-col gap-2 p-2">
      <div className="flex flex-col gap-1.5">
        <span className="text-[13px] text-fg-secondary">Y</span>
        <SearchSelect label="Y axis" options={yOptions} value={y ?? ""} onValue={(next) => onConfig({ y: next })} placeholder="Search metrics and parameters" emptyMessage="No numeric field" align="start" />
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="text-[13px] text-fg-secondary">X</span>
        <SearchSelect label="X axis" options={xOptions} value={x} onValue={(next) => onConfig({ x: next })} align="start" />
      </div>
      <Field label="Y scale" group>
        <SegmentedControl label="Y scale" options={SCALE_OPTIONS} value={config.logY ? "log" : "linear"} onValue={(value) => onConfig({ logY: value === "log" })} />
      </Field>
      {numericX ? (
        <Field label="X scale" group>
          <SegmentedControl label="X scale" options={SCALE_OPTIONS} value={config.logX ? "log" : "linear"} onValue={(value) => onConfig({ logX: value === "log" })} />
        </Field>
      ) : null}
    </div>
  );

  return (
    <Card className="group/card flex flex-col">
      <div className="flex h-card-head flex-none items-center justify-between gap-2 border-b border-line pr-1 pl-3">
        <button type="button" onClick={() => setSettingsOpen(true)} className="min-w-0 truncate text-left text-[13px] font-medium text-fg hover:text-accent">
          {y === null ? "New plot" : `${axisLabel(y)} by ${axisLabel(x)}`}
        </button>
        <div className={settingsOpen ? "flex shrink-0 items-center" : "flex shrink-0 items-center opacity-0 group-hover/card:opacity-100 focus-within:opacity-100"}>
          <IconButton label="Reset zoom" size="sm" onClick={resetZoom}>
            <RotateCcw size={13} />
          </IconButton>
          <Popover
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            title="Plot settings"
            width={300}
            trigger={
              <IconButton label="Plot settings" size="sm" active={settingsOpen} onClick={() => setSettingsOpen(!settingsOpen)}>
                <Settings2 size={13} />
              </IconButton>
            }
          >
            {settings}
          </Popover>
          <IconButton label="Remove plot" size="sm" onClick={onRemove}>
            <X size={13} />
          </IconButton>
        </div>
      </div>
      {y === null || built === null || layout === null ? (
        <EmptyState compact icon={<span />} title="Choose what to plot" description="Open the settings to pick a metric for Y and, optionally, a parameter for X.">
          <IconButton label="Plot settings" onClick={() => setSettingsOpen(true)}>
            <Settings2 size={14} />
          </IconButton>
        </EmptyState>
      ) : (
        <div className="bg-plot" style={{ height: PLOT_HEIGHT }}>
          <Plot
            data={built.traces}
            layout={layout}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
            onInitialized={(_, graph) => {
              graphRef.current = graph;
            }}
            onUpdate={(_, graph) => {
              graphRef.current = graph;
            }}
            onHover={(event) => {
              const meta = (event.points[0].data as { meta?: { runId: string } }).meta;
              if (meta !== undefined) setCursor({ x: null, axis: "step", source: `compare:${config.id}`, runId: meta.runId, pointerY: 0 });
            }}
            onUnhover={clearCursor}
          />
        </div>
      )}
    </Card>
  );
}
