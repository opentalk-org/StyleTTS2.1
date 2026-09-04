import type { GlobalPlotSettings, PlotSettings, XAxis } from "@/shared/types";
import { Button, Field, Range, SearchSelect, SegmentedControl } from "@/shared/ui";

const RENDER_OPTIONS = [
  { value: "line" as const, label: "Line" },
  { value: "scatter" as const, label: "Points" },
  { value: "line-scatter" as const, label: "Both" },
];

const SMOOTHING_OPTIONS = [
  { value: "inherit" as const, label: "Global", title: "Use the smoothing from the toolbar" },
  { value: "none" as const, label: "Off" },
  { value: "ema" as const, label: "EMA", title: "Exponential moving average" },
  { value: "mean" as const, label: "Mean", title: "Rolling mean" },
];

const SCALE_OPTIONS = [
  { value: "linear" as const, label: "Linear" },
  { value: "log" as const, label: "Log" },
];

const X_AXIS_OPTIONS: { value: "inherit" | XAxis; label: string; title?: string }[] = [
  { value: "inherit", label: "Global", title: "Use the x axis from the toolbar" },
  { value: "step", label: "Step" },
  { value: "lineage", label: "Lineage", title: "Step continued across the runs this one resumed from" },
  { value: "relative", label: "Time", title: "Seconds since the run started" },
  { value: "wall", label: "Wall", title: "Wall-clock time" },
];

const X_AXIS_LABELS: Record<XAxis, string> = { step: "step", lineage: "lineage step", relative: "time", wall: "wall clock" };

interface ChartSettingsFormProps {
  settings: PlotSettings;
  global: GlobalPlotSettings;
  onChange: (patch: Partial<PlotSettings>) => void;
  onReset?: () => void;
  /** When given, the form also lets the user switch which metric is shown. */
  metrics?: string[];
  metric?: string;
  onMetric?: (name: string) => void;
  /** False when the query does not return wall/relative time columns. */
  hasTime?: boolean;
}

/** Immediate-apply settings for one chart; used in the expanded chart dialog. */
export function ChartSettingsForm({
  settings,
  global,
  onChange,
  onReset,
  metrics,
  metric,
  onMetric,
  hasTime = true,
}: ChartSettingsFormProps) {
  return (
    <div className="flex flex-col gap-1 p-2">
      {metrics === undefined || metric === undefined || onMetric === undefined ? null : (
        <div className="flex flex-col gap-1.5 py-1">
          <span className="text-[13px] text-fg-secondary">Metric (y)</span>
          <SearchSelect
            label="Metric"
            options={metrics.map((name) => ({ value: name, label: name }))}
            value={metric}
            onValue={onMetric}
            placeholder="Search metrics"
            align="start"
          />
        </div>
      )}
      <div className="flex flex-col gap-1.5 py-1">
        <span className="text-[13px] text-fg-secondary">X axis</span>
        <SegmentedControl
          fill
          label="X axis"
          options={X_AXIS_OPTIONS}
          value={settings.xAxis}
          onValue={(xAxis) => onChange({ xAxis })}
          disabled={!hasTime}
        />
        <span className="text-xs text-fg-muted">
          {!hasTime
            ? "The query has no wall/rel columns, so only step is available"
            : settings.xAxis === "inherit"
              ? `Toolbar: ${X_AXIS_LABELS[global.xAxis]}`
              : ""}
        </span>
      </div>
      <Field label="X scale" group>
        <SegmentedControl label="X scale" options={SCALE_OPTIONS} value={settings.xScale} onValue={(xScale) => onChange({ xScale })} />
      </Field>
      <Field label="Y scale" group>
        <SegmentedControl label="Y scale" options={SCALE_OPTIONS} value={settings.yScale} onValue={(yScale) => onChange({ yScale })} />
      </Field>
      <Field label="Display" group>
        <SegmentedControl label="Display" options={RENDER_OPTIONS} value={settings.renderMode} onValue={(renderMode) => onChange({ renderMode })} />
      </Field>
      <div className="flex flex-col gap-1.5 py-1">
        <span className="text-[13px] text-fg-secondary">Smoothing</span>
        <SegmentedControl
          fill
          label="Smoothing"
          options={SMOOTHING_OPTIONS}
          value={settings.smoothing}
          onValue={(smoothing) =>
            onChange({ smoothing, smoothingValue: smoothing === "mean" ? 10 : smoothing === "ema" ? 0.75 : settings.smoothingValue })
          }
        />
        {settings.smoothing === "inherit" ? (
          <span className="text-xs text-fg-muted">
            {global.smoothing > 0 ? `Toolbar: EMA ${global.smoothing.toFixed(2)}` : "Toolbar: off"}
          </span>
        ) : null}
      </div>
      {settings.smoothing === "ema" ? (
        <Field label="Weight" value={settings.smoothingValue.toFixed(2)}>
          <Range min={0.05} max={0.99} step={0.01} value={settings.smoothingValue} onValue={(smoothingValue) => onChange({ smoothingValue })} className="w-32" />
        </Field>
      ) : null}
      {settings.smoothing === "mean" ? (
        <Field label="Window" value={settings.smoothingValue.toFixed(0)}>
          <Range min={2} max={100} step={1} value={settings.smoothingValue} onValue={(smoothingValue) => onChange({ smoothingValue })} className="w-32" />
        </Field>
      ) : null}
      {onReset === undefined ? null : (
        <div className="mt-1 border-t border-line pt-2">
          <Button size="sm" variant="ghost" onClick={onReset}>
            Reset to defaults
          </Button>
        </div>
      )}
    </div>
  );
}
