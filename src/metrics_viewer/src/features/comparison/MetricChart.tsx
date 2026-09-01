import { EyeOff, RotateCcw, Settings2, X } from "lucide-react";
import Plotly from "plotly.js-basic-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import { useMemo, useState, type ReactNode } from "react";

import { runColor } from "@/shared/chart";
import type { PlotSettings, Run } from "@/shared/types";
import {
  Button,
  Card,
  CardHeader,
  Checkbox,
  cn,
  Field,
  GroupLabel,
  IconButton,
  Modal,
  Range,
  SegmentedControl,
} from "@/shared/ui";
import { buildTraces, cursorXAt, plotLayout, type Plot as PlotData } from "./logic";

const Plot = createPlotlyComponent(Plotly);

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  displayModeBar: false,
  scrollZoom: true,
  doubleClick: "reset",
} as const;

const CARD_PLOT_HEIGHT = 280;

interface MetricChartProps {
  plot: PlotData;
  runs: Run[];
  settings: PlotSettings;
  runColors: Record<string, string>;
  xLabel: string;
  cursorIndex: number | null;
  onCursorIndex: (index: number | null) => void;
  onChange: (patch: Partial<PlotSettings>) => void;
  onReset: () => void;
  onHide: () => void;
}

export function MetricChart({
  plot,
  runs,
  settings,
  runColors,
  xLabel,
  cursorIndex,
  onCursorIndex,
  onChange,
  onReset,
  onHide,
}: MetricChartProps) {
  const [draft, setDraft] = useState<PlotSettings | null>(null);
  const [hiddenRuns, setHiddenRuns] = useState<Set<string>>(new Set());
  const [plotKey, setPlotKey] = useState(0);

  // Traces are the expensive part and do not depend on the hovered point, so they
  // stay stable while the cursor moves: Plotly then only re-draws the cursor line.
  const traces = useMemo(
    () => buildTraces(plot, runs, hiddenRuns, settings, runColors),
    [plot, runs, hiddenRuns, settings, runColors],
  );
  const cursorX = cursorXAt(plot, runs, cursorIndex);
  const layout = useMemo(() => plotLayout(settings, cursorX, CARD_PLOT_HEIGHT), [settings, cursorX]);

  function toggleRun(id: string) {
    setHiddenRuns(toggleSet(hiddenRuns, id));
  }

  function closeSettings() {
    const dirty = draft !== null && JSON.stringify(draft) !== JSON.stringify(settings);
    if (dirty && !window.confirm("Discard the changes to this plot?")) return;
    setDraft(null);
  }

  return (
    <Card>
      <CardHeader className="h-14">
        <div className="flex min-w-0 flex-col gap-1">
          <h3 className="m-0 truncate text-sm font-semibold tracking-tight text-fg">{plot.name}</h3>
          <span className="truncate font-mono text-[10px] text-fg-muted">
            {xLabel} · {plot.series.length} series · {plot.pointCount.toLocaleString()} points
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <IconButton label="Reset zoom" size="sm" onClick={() => setPlotKey((value) => value + 1)}>
            <RotateCcw size={13} />
          </IconButton>
          <IconButton label="Hide plot in this view" size="sm" onClick={onHide}>
            <EyeOff size={13} />
          </IconButton>
          <IconButton
            label="Plot settings"
            size="sm"
            active={draft !== null}
            onClick={() => setDraft({ ...settings })}
          >
            <Settings2 size={14} />
          </IconButton>
        </div>
      </CardHeader>
      <Plot
        key={plotKey}
        data={traces}
        layout={layout}
        config={PLOT_CONFIG}
        useResizeHandler
        className="w-full"
        onHover={(event) => onCursorIndex(event.points[0].pointIndex)}
        onUnhover={() => onCursorIndex(null)}
      />
      {settings.showLegend && runs.length > 1 ? (
        <RunLegend runs={runs} runColors={runColors} hiddenRuns={hiddenRuns} onToggle={toggleRun} />
      ) : null}
      {draft === null ? null : (
        <PlotSettingsDialog
          name={plot.name}
          draft={draft}
          plot={plot}
          runs={runs}
          runColors={runColors}
          hiddenRuns={hiddenRuns}
          onToggleRun={toggleRun}
          onDraft={(patch) => setDraft({ ...draft, ...patch })}
          onCancel={closeSettings}
          onReset={() => {
            onReset();
            setDraft(null);
          }}
          onSave={() => {
            onChange(draft);
            setDraft(null);
          }}
        />
      )}
    </Card>
  );
}

interface PlotSettingsDialogProps {
  name: string;
  draft: PlotSettings;
  plot: PlotData;
  runs: Run[];
  runColors: Record<string, string>;
  hiddenRuns: Set<string>;
  onToggleRun: (id: string) => void;
  onDraft: (patch: Partial<PlotSettings>) => void;
  onCancel: () => void;
  onReset: () => void;
  onSave: () => void;
}

/** Full-viewport editor: live plot on the left, settings on the right. */
function PlotSettingsDialog({
  name,
  draft,
  plot,
  runs,
  runColors,
  hiddenRuns,
  onToggleRun,
  onDraft,
  onCancel,
  onReset,
  onSave,
}: PlotSettingsDialogProps) {
  const traces = useMemo(
    () => buildTraces(plot, runs, hiddenRuns, draft, runColors),
    [plot, runs, hiddenRuns, draft, runColors],
  );
  const layout = useMemo(() => plotLayout(draft, null), [draft]);

  return (
    <Modal open onClose={onCancel} label={`${name} settings`} className="max-w-[1400px]">
      <header className="flex h-14 flex-none items-center justify-between gap-3 border-b border-line px-4">
        <div className="flex min-w-0 flex-col gap-1">
          <GroupLabel>Plot settings</GroupLabel>
          <h2 className="m-0 truncate text-base leading-tight font-semibold tracking-tight text-fg">
            {name}
          </h2>
        </div>
        <IconButton label="Close settings" onClick={onCancel}>
          <X size={15} />
        </IconButton>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[1fr_420px]">
        <section aria-label="Preview" className="flex min-h-0 min-w-0 flex-col bg-base">
          <div className="min-h-0 flex-1 p-2">
            <Plot
              data={traces}
              layout={layout}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
            />
          </div>
          {runs.length > 1 ? (
            <RunLegend
              runs={runs}
              runColors={runColors}
              hiddenRuns={hiddenRuns}
              onToggle={onToggleRun}
            />
          ) : null}
        </section>

        <section
          aria-label="Settings"
          className="flex min-h-0 flex-col border-t border-line lg:border-t-0 lg:border-l"
        >
          <div className="min-h-0 flex-1 overflow-auto p-4">
            <SettingsForm settings={draft} onChange={onDraft} />
          </div>
          <footer className="flex flex-none items-center justify-between gap-2 border-t border-line p-3">
            <Button variant="ghost" onClick={onReset}>
              Reset to defaults
            </Button>
            <div className="flex items-center gap-2">
              <Button variant="ghost" onClick={onCancel}>
                Cancel
              </Button>
              <Button variant="primary" onClick={onSave}>
                Save changes
              </Button>
            </div>
          </footer>
        </section>
      </div>
    </Modal>
  );
}

const RENDER_OPTIONS = [
  { value: "line" as const, label: "Line" },
  { value: "scatter" as const, label: "Points" },
  { value: "line-scatter" as const, label: "Both", title: "Line + points" },
];

const SMOOTHING_OPTIONS = [
  { value: "none" as const, label: "None" },
  { value: "ema" as const, label: "EMA", title: "Exponential moving average" },
  { value: "mean" as const, label: "Rolling", title: "Rolling mean" },
];

/**
 * A scale is one bit per axis, so it lives next to that axis rather than in its own
 * radio row. The label is the current state — no second element to read it from.
 */
function ScaleToggle({
  axis,
  value,
  onValue,
}: {
  axis: "X" | "Y";
  value: "linear" | "log";
  onValue: (value: "linear" | "log") => void;
}) {
  const isLog = value === "log";
  return (
    <button
      type="button"
      aria-pressed={isLog}
      aria-label={`${axis} scale: ${isLog ? "logarithmic" : "linear"}`}
      title={`${axis} scale is ${isLog ? "logarithmic" : "linear"} — click to switch`}
      onClick={() => onValue(isLog ? "linear" : "log")}
      className={cn(
        "h-8 w-11 shrink-0 rounded-lg border font-mono text-[11px]",
        "transition-[background-color,border-color,color] duration-150 ease-out active:scale-[0.98]",
        isLog
          ? "border-accent-border bg-accent-surface text-accent-bright"
          : "border-line text-fg-muted hover:border-line-hover hover:text-fg-secondary",
      )}
    >
      {isLog ? "log" : "lin"}
    </button>
  );
}

/** Label above the control, for controls that use the panel's full width. */
function StackedField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5 py-1.5">
      <span className="text-sm text-fg-secondary">{label}</span>
      {children}
    </div>
  );
}

function SettingsForm({
  settings,
  onChange,
}: {
  settings: PlotSettings;
  onChange: (patch: Partial<PlotSettings>) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      <fieldset className="m-0 flex flex-col border-0 p-0">
        <legend className="mb-1 p-0">
          <GroupLabel>Axes</GroupLabel>
        </legend>
        <p className="m-0 mb-1 text-xs leading-relaxed text-fg-muted">
          What each axis plots is decided by the query above the charts; only the scale is
          set here.
        </p>
        <Field label="X scale" group>
          <ScaleToggle axis="X" value={settings.xScale} onValue={(xScale) => onChange({ xScale })} />
        </Field>
        <Field label="Y scale" group>
          <ScaleToggle axis="Y" value={settings.yScale} onValue={(yScale) => onChange({ yScale })} />
        </Field>
      </fieldset>

      <fieldset className="m-0 flex flex-col border-0 p-0">
        <legend className="mb-1 p-0">
          <GroupLabel>Series</GroupLabel>
        </legend>
        <StackedField label="Display">
          <SegmentedControl
            fill
            label="Display"
            options={RENDER_OPTIONS}
            value={settings.renderMode}
            onValue={(renderMode) => onChange({ renderMode })}
          />
        </StackedField>
        <StackedField label="Smoothing">
          <SegmentedControl
            fill
            label="Smoothing"
            options={SMOOTHING_OPTIONS}
            value={settings.smoothing}
            onValue={(smoothing) => onChange({ smoothing })}
          />
        </StackedField>
        {settings.smoothing === "ema" ? (
          <Field label="EMA weight" value={settings.smoothingValue.toFixed(2)}>
            <Range
              min={0.05}
              max={0.95}
              step={0.05}
              value={settings.smoothingValue}
              onValue={(smoothingValue) => onChange({ smoothingValue })}
            />
          </Field>
        ) : null}
        {settings.smoothing === "mean" ? (
          <Field label="Mean window" value={settings.smoothingValue.toFixed(0)}>
            <Range
              min={2}
              max={50}
              step={1}
              value={settings.smoothingValue}
              onValue={(smoothingValue) => onChange({ smoothingValue })}
            />
          </Field>
        ) : null}
        {settings.smoothing === "none" ? null : (
          <>
            <Field label="Raw opacity" value={settings.rawOpacity.toFixed(2)}>
              <Range
                min={0}
                max={1}
                step={0.05}
                value={settings.rawOpacity}
                onValue={(rawOpacity) => onChange({ rawOpacity })}
              />
            </Field>
            <Field label="Smooth opacity" value={settings.smoothOpacity.toFixed(2)}>
              <Range
                min={0.1}
                max={1}
                step={0.05}
                value={settings.smoothOpacity}
                onValue={(smoothOpacity) => onChange({ smoothOpacity })}
              />
            </Field>
          </>
        )}
        <Checkbox
          className="-mx-2"
          checked={settings.showLegend}
          onChange={(event) => onChange({ showLegend: event.target.checked })}
        >
          Show legend
        </Checkbox>
      </fieldset>
    </div>
  );
}

function RunLegend({
  runs,
  runColors,
  hiddenRuns,
  onToggle,
}: {
  runs: Run[];
  runColors: Record<string, string>;
  hiddenRuns: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="flex min-h-10 flex-none items-center gap-1.5 overflow-x-auto border-t border-line bg-inset px-2.5 py-1.5">
      {runs.map((run, position) => {
        const hidden = hiddenRuns.has(run.id);
        const color = runColor(run.id, position, runColors);
        return (
          <button
            key={run.id}
            type="button"
            aria-pressed={!hidden}
            onClick={() => onToggle(run.id)}
            className={cn(
              "flex h-6 shrink-0 items-center gap-1.5 rounded-md border px-2 font-mono text-[10px] whitespace-nowrap",
              "transition-[color,border-color,opacity] duration-150 ease-out",
              hidden
                ? "border-line text-fg-muted line-through opacity-50"
                : "border-line text-fg-secondary hover:border-line-hover hover:text-fg",
            )}
          >
            <span
              aria-hidden
              className="size-1.5 shrink-0 rounded-full"
              style={{
                background: hidden ? "transparent" : color,
                boxShadow: `inset 0 0 0 1px ${color}`,
              }}
            />
            {run.name}
          </button>
        );
      })}
    </div>
  );
}

function toggleSet(values: Set<string>, value: string): Set<string> {
  const next = new Set(values);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}
