import { Columns3, Eraser, Settings2, Waypoints, X } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { useCursorStore } from "@/shared/cursor";
import type { ComparePlotConfig, Run } from "@/shared/types";
import { Button, Caption, Card, EmptyState, Field, IconButton, Popover, SearchOptionList, SearchSelect, Tooltip, type SearchOption } from "@/shared/ui";

import { cellText, isNumeric } from "./logic";
import {
  buildAxes,
  colorScaleRange,
  compactNumber,
  GRADIENT_CSS,
  lineColors,
  normalizeBox,
  passesBrushes,
  polylineHitsBox,
  runPositions,
  segments,
  type Box,
  type Brushes,
  type ParallelAxis,
} from "./parallel";

const PLOT_HEIGHT = 320;
const MIN_LABEL_WIDTH = 56;
const MAX_LABEL_WIDTH = 160;
/** Breathing room between an axis label or tick and the neighbouring axis. */
const LABEL_GAP = 10;
const TICK_CHAR_WIDTH = 6;
const MIN_TICK_CHARS = 4;
const PADDING = { top: 34, right: 40, bottom: 26, left: 76 };
/** Pointer travel below this many pixels is a click, which clears the axis brush. */
const CLICK_SLOP = 4;
const BRUSH_HIT_WIDTH = 22;
const RUN_COLOR = "run";

interface Brush {
  column: string;
  /** Pixel y of the drag start and of the pointer, before normalization. */
  from: number;
  to: number;
}

interface Reorder {
  column: string;
  index: number;
  x: number;
}

interface ParallelPlotProps {
  runs: Run[];
  runColors: Record<string, string>;
  /** The compare table's columns; the axes follow them until the plot picks its own. */
  columns: string[];
  columnOptions: SearchOption[];
  config: ComparePlotConfig;
  onConfig: (patch: Partial<ComparePlotConfig>) => void;
  onRemove: () => void;
}

/**
 * Parallel coordinates: one line per run across the chosen parameters and metrics, as in
 * W&B. Dragging on an axis brushes a range and dims every run outside it; dragging an axis
 * label reorders the axes. Hovering a line highlights that run in the run list.
 */
export function ParallelPlot({ runs, runColors, columns, columnOptions, config, onConfig, onRemove }: ParallelPlotProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [axesOpen, setAxesOpen] = useState(false);
  const [brushes, setBrushes] = useState<Brushes>({});
  const [brush, setBrush] = useState<Brush | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  /**
   * A pointerdown and the pointerup that ends it can be handled before React has
   * re-rendered, so the drag in progress is kept in refs and mirrored into state only for
   * drawing. Reading it from state loses quick drags and clicks.
   */
  const brushRef = useRef<Brush | null>(null);
  const boxRef = useRef<Box | null>(null);
  const reorderRef = useRef<Reorder | null>(null);
  /** Run ids picked by the last box selection, or null while nothing is boxed. */
  const [boxed, setBoxed] = useState<string[] | null>(null);
  const [reorder, setReorder] = useState<Reorder | null>(null);
  const setCursor = useCursorStore((state) => state.set);
  const clearCursor = useCursorStore((state) => state.clear);
  const hoveredRunId = useCursorStore((state) => state.runId);

  const known = useMemo(() => new Set(columnOptions.map((option) => option.value)), [columnOptions]);
  const dims = useMemo(() => (config.dims ?? columns).filter((column) => known.has(column)), [config.dims, columns, known]);
  const colorBy = config.colorBy !== null && known.has(config.colorBy) ? config.colorBy : null;

  const axes = useMemo(() => buildAxes(runs, dims), [runs, dims]);
  const lines = useMemo(() => runPositions(runs, axes), [runs, axes]);
  const colors = useMemo(() => lineColors(runs, colorBy, runColors), [runs, colorBy, runColors]);
  const scale = useMemo(() => (colorBy === null ? null : colorScaleRange(runs, colorBy)), [runs, colorBy]);
  const numericOptions = useMemo(() => columnOptions.filter((option) => isNumeric(runs, option.value)), [columnOptions, runs]);

  // Brushes are per axis and a box selection is drawn against the old lines, so both are
  // meaningless once the axes change — keeping them silently empties the plot.
  const axisKey = dims.join("\u0000");
  useEffect(() => {
    setBrushes({});
    setBoxed(null);
  }, [axisKey]);

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (container === null) return;
    setWidth(container.clientWidth);
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const innerHeight = PLOT_HEIGHT - PADDING.top - PADDING.bottom;
  const span = Math.max(0, width - PADDING.left - PADDING.right);
  // Everything that has to fit between two axes — labels and tick text — is sized off this.
  const slot = axes.length <= 1 ? span : span / (axes.length - 1);
  const axisX = (index: number) => PADDING.left + (axes.length <= 1 ? span / 2 : (index * span) / (axes.length - 1));
  const yOf = (position: number) => PADDING.top + (1 - position) * innerHeight;
  const positionOf = (pixelY: number) => Math.min(1, Math.max(0, 1 - (pixelY - PADDING.top) / innerHeight));

  const boxedIds = useMemo(() => (boxed === null ? null : new Set(boxed)), [boxed]);
  const active = useMemo(
    () =>
      lines.filter(
        (line) => passesBrushes(axes, line.positions, brushes) && (boxedIds === null || boxedIds.has(line.run.id)),
      ),
    [lines, axes, brushes, boxedIds],
  );
  const activeIds = useMemo(() => new Set(active.map((line) => line.run.id)), [active]);
  const hovered = hoveredRunId === null ? null : lines.find((line) => line.run.id === hoveredRunId) ?? null;
  const filtered = Object.keys(brushes).length > 0 || boxed !== null;
  const labelWidth = Math.max(MIN_LABEL_WIDTH, Math.min(MAX_LABEL_WIDTH, slot - LABEL_GAP));
  const maxTickChars = Math.max(MIN_TICK_CHARS, Math.floor((slot - LABEL_GAP) / TICK_CHAR_WIDTH));

  function localPoint(event: React.PointerEvent): { x: number; y: number } {
    const area = svgRef.current?.getBoundingClientRect();
    return area === undefined ? { x: 0, y: 0 } : { x: event.clientX - area.left, y: event.clientY - area.top };
  }

  function startBrush(event: React.PointerEvent, column: string) {
    // The axis owns this drag; without this the plot would also start a box selection.
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const y = localPoint(event).y;
    brushRef.current = { column, from: y, to: y };
    setBrush(brushRef.current);
  }

  function moveBrush(event: React.PointerEvent) {
    event.stopPropagation();
    const current = brushRef.current;
    if (current === null) return;
    brushRef.current = { ...current, to: localPoint(event).y };
    setBrush(brushRef.current);
  }

  function endBrush(event: React.PointerEvent) {
    event.stopPropagation();
    const drag = brushRef.current;
    if (drag === null) return;
    brushRef.current = null;
    setBrush(null);
    setBrushes((current) => {
      const next = { ...current };
      if (Math.abs(drag.to - drag.from) < CLICK_SLOP) delete next[drag.column];
      else next[drag.column] = [positionOf(Math.max(drag.from, drag.to)), positionOf(Math.min(drag.from, drag.to))];
      return next;
    });
  }

  /** Lines as drawn, in pixels, for hit-testing a box selection. */
  function pixelSegments(positions: (number | null)[]): [number, number][][] {
    return segments(positions).map((segment) => segment.map(([index, position]) => [axisX(index), yOf(position)] as [number, number]));
  }

  function startBox(event: React.PointerEvent) {
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = localPoint(event);
    boxRef.current = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
    setBox(boxRef.current);
  }

  function moveBox(event: React.PointerEvent) {
    const current = boxRef.current;
    if (current === null) return;
    const point = localPoint(event);
    boxRef.current = { ...current, x1: point.x, y1: point.y };
    setBox(boxRef.current);
  }

  /** A real drag keeps the lines it touches; a click on the background clears the selection. */
  function endBox() {
    const drag = boxRef.current;
    if (drag === null) return;
    boxRef.current = null;
    setBox(null);
    const area = normalizeBox(drag);
    if (area.x1 - area.x0 < CLICK_SLOP && area.y1 - area.y0 < CLICK_SLOP) {
      setBoxed(null);
      return;
    }
    const hit = lines.filter(({ positions }) => pixelSegments(positions).some((segment) => polylineHitsBox(segment, area)));
    setBoxed(hit.map(({ run }) => run.id));
  }

  function startReorder(event: React.PointerEvent, column: string, index: number) {
    event.currentTarget.setPointerCapture(event.pointerId);
    reorderRef.current = { column, index, x: axisX(index) };
    setReorder(reorderRef.current);
  }

  function moveReorder(event: React.PointerEvent) {
    const current = reorderRef.current;
    if (current === null) return;
    const area = containerRef.current?.getBoundingClientRect();
    if (area === undefined) return;
    reorderRef.current = { ...current, x: event.clientX - area.left };
    setReorder(reorderRef.current);
  }

  function endReorder() {
    const drag = reorderRef.current;
    if (drag === null) return;
    reorderRef.current = null;
    setReorder(null);
    const target = nearestIndex(drag.x, axes.length, axisX);
    if (target === drag.index) return;
    const next = dims.filter((column) => column !== drag.column);
    next.splice(target, 0, drag.column);
    onConfig({ dims: next });
  }

  const settings = (
    <div className="flex flex-col gap-2 p-2">
      <Field label="Colour by" group>
        <SearchSelect
          label="Colour by"
          options={[{ value: RUN_COLOR, label: "Run colour", group: "Run" }, ...numericOptions]}
          value={colorBy ?? RUN_COLOR}
          onValue={(next) => onConfig({ colorBy: next === RUN_COLOR ? null : next })}
          align="start"
        />
      </Field>
      {config.dims === null ? null : (
        <Button variant="ghost" size="sm" onClick={() => onConfig({ dims: null })}>
          Follow table columns
        </Button>
      )}
    </div>
  );

  return (
    <Card className="group/card col-span-full flex flex-col">
      <div className="flex h-card-head flex-none items-center justify-between gap-2 border-b border-line pr-1 pl-3">
        <button type="button" onClick={() => setSettingsOpen(true)} className="min-w-0 truncate text-left text-[13px] font-medium text-fg hover:text-accent">
          Parallel coordinates
          <span className="ml-2 font-mono text-[11px] text-fg-muted">
            {filtered ? `${active.length}/${lines.length}` : lines.length} runs
          </span>
        </button>
        <div className={settingsOpen || axesOpen ? "flex shrink-0 items-center" : "flex shrink-0 items-center opacity-0 group-hover/card:opacity-100 focus-within:opacity-100"}>
          <Popover
            open={axesOpen}
            onClose={() => setAxesOpen(false)}
            align="end"
            width={320}
            panelClassName="[&>div]:p-0"
            trigger={
              <IconButton label="Axes" size="sm" active={axesOpen} onClick={() => setAxesOpen(!axesOpen)}>
                <Columns3 size={13} />
              </IconButton>
            }
          >
            <SearchOptionList
              multiple
              options={columnOptions}
              selected={dims}
              placeholder="Search parameters and metrics"
              emptyMessage="No column matches"
              onSelect={(id) => onConfig({ dims: dims.includes(id) ? dims.filter((column) => column !== id) : [...dims, id] })}
            />
          </Popover>
          {filtered ? (
            <IconButton label="Clear selection" size="sm" onClick={() => { setBrushes({}); setBoxed(null); }}>
              <Eraser size={13} />
            </IconButton>
          ) : null}
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
      {axes.length < 2 ? (
        <EmptyState compact icon={<Waypoints />} title="Pick at least two axes" description="The axes follow the table's columns; pick them here to give this plot its own.">
          <IconButton label="Axes" onClick={() => setAxesOpen(true)}>
            <Columns3 size={14} />
          </IconButton>
        </EmptyState>
      ) : (
        <div ref={containerRef} className="relative bg-plot" style={{ height: PLOT_HEIGHT }}>
          {/* Before the container is measured every axis would sit at the same x, stacked. */}
          {width === 0 ? null : (
          <>
          <svg
            ref={svgRef}
            width={width}
            height={PLOT_HEIGHT}
            className="block cursor-crosshair touch-none select-none"
            onPointerLeave={clearCursor}
            onPointerDown={startBox}
            onPointerMove={moveBox}
            onPointerUp={endBox}
          >
            <g>
              {lines.map(({ run, positions }) => {
                const dimmed = !activeIds.has(run.id);
                const lit = hoveredRunId === null ? null : hoveredRunId === run.id;
                if (lit === true) return null;
                return segments(positions).map((segment, index) => (
                  <polyline
                    key={`${run.id}-${index}`}
                    points={pointsOf(segment, axisX, yOf)}
                    fill="none"
                    stroke={colors[run.id]}
                    strokeWidth={1.5}
                    strokeLinejoin="round"
                    opacity={dimmed ? 0.07 : lit === false ? 0.18 : 0.85}
                  />
                ));
              })}
              {hovered === null
                ? null
                : segments(hovered.positions).map((segment, index) => (
                    <polyline
                      key={index}
                      points={pointsOf(segment, axisX, yOf)}
                      fill="none"
                      stroke={colors[hovered.run.id]}
                      strokeWidth={2.5}
                      strokeLinejoin="round"
                    />
                  ))}
            </g>
            <g>
              {lines.map(({ run, positions }) =>
                segments(positions).map((segment, index) => (
                  <polyline
                    key={`${run.id}-${index}`}
                    points={pointsOf(segment, axisX, yOf)}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={10}
                    className="cursor-pointer"
                    onPointerEnter={() =>
                      setCursor({ x: null, axis: "step", source: `parallel:${config.id}`, runId: run.id, pointerX: 0, pointerY: 0 })
                    }
                  />
                )),
              )}
            </g>
            {axes.map((axis, index) => (
              <AxisMarks
                key={axis.column}
                axis={axis}
                maxTickChars={maxTickChars}
                x={axisX(index)}
                top={PADDING.top}
                height={innerHeight}
                yOf={yOf}
                brush={brushes[axis.column]}
                dragging={brush?.column === axis.column ? brush : null}
                onPointerDown={(event) => startBrush(event, axis.column)}
                onPointerMove={moveBrush}
                onPointerUp={endBrush}
              />
            ))}
          {box === null ? null : (
              <rect
                {...rectOf(normalizeBox(box))}
                fill="var(--accent)"
                fillOpacity={0.1}
                stroke="var(--accent)"
                strokeDasharray="4 3"
                pointerEvents="none"
              />
            )}
          </svg>
          {lines.length === 0 || active.length > 0 ? null : (
            <div className="pointer-events-none absolute inset-0 grid place-items-center">
              <span className="rounded-md border border-line bg-raised/95 px-2 py-1 text-xs text-fg-muted">
                No run matches the selection
              </span>
            </div>
          )}
          <div className="pointer-events-none absolute inset-0">
            {axes.map((axis, index) => (
              <div
                key={axis.column}
                className="pointer-events-auto absolute -translate-x-1/2 overflow-hidden cursor-grab active:cursor-grabbing"
                style={{
                  left: reorder?.column === axis.column ? reorder.x : axisX(index),
                  top: 8,
                  width: labelWidth,
                  opacity: reorder?.column === axis.column ? 0.75 : 1,
                }}
                onPointerDown={(event) => startReorder(event, axis.column, index)}
                onPointerMove={moveReorder}
                onPointerUp={endReorder}
              >
                <Tooltip content={<span className="font-mono">{axis.column}</span>} className="w-full min-w-0">
                  <Caption className="block w-full truncate text-center text-fg-secondary">
                    {axis.label}
                    {axis.log ? <span className="ml-1 text-fg-muted">log</span> : null}
                  </Caption>
                </Tooltip>
              </div>
            ))}
          </div>
          {hovered === null ? null : (
            <div className="pointer-events-none absolute top-8 left-2 max-w-64 rounded-md border border-line bg-raised/95 px-2 py-1.5 shadow-popover">
              <div className="flex items-center gap-1.5">
                <span className="size-2.5 shrink-0 rounded-full" style={{ background: colors[hovered.run.id] }} />
                <span className="truncate text-xs font-medium text-fg">{hovered.run.name}</span>
              </div>
              {axes.map((axis) => (
                <div key={axis.column} className="flex justify-between gap-3 font-mono text-[11px] text-fg-muted">
                  <span className="truncate">{axis.label}</span>
                  <span className="shrink-0 tabular-nums text-fg-secondary">{cellText(hovered.run, axis.column)}</span>
                </div>
              ))}
            </div>
          )}
          {scale === null || colorBy === null ? null : (
            <div className="pointer-events-none absolute right-3 bottom-1 flex items-center gap-1.5">
              <span className="font-mono text-[10px] text-fg-muted">{compactNumber(scale.low)}</span>
              <span className="h-1.5 w-20 rounded-full" style={{ background: GRADIENT_CSS }} />
              <span className="font-mono text-[10px] text-fg-muted">{compactNumber(scale.high)}</span>
            </div>
          )}
          </>
          )}
        </div>
      )}
    </Card>
  );
}

interface AxisMarksProps {
  axis: ParallelAxis;
  /** Tick text is clipped to what fits between this axis and the next. */
  maxTickChars: number;
  x: number;
  top: number;
  height: number;
  yOf: (position: number) => number;
  brush: [number, number] | undefined;
  dragging: Brush | null;
  onPointerDown: (event: React.PointerEvent) => void;
  onPointerMove: (event: React.PointerEvent) => void;
  onPointerUp: (event: React.PointerEvent) => void;
}

/** One vertical axis: its line, tick labels, the committed brush and the one being dragged. */
function AxisMarks({ axis, maxTickChars, x, top, height, yOf, brush, dragging, onPointerDown, onPointerMove, onPointerUp }: AxisMarksProps) {
  const drag = dragging === null ? null : { y: Math.min(dragging.from, dragging.to), height: Math.abs(dragging.to - dragging.from) };
  return (
    <g>
      <line x1={x} x2={x} y1={top} y2={top + height} stroke="var(--strong)" strokeWidth={1} />
      {axis.ticks.map((tick) => (
        <text
          key={`${tick.position}-${tick.label}`}
          x={x - 8}
          y={yOf(tick.position)}
          textAnchor="end"
          dominantBaseline="middle"
          fill="var(--fg-muted)"
          className="font-mono text-[10px]"
        >
          {clip(tick.label, maxTickChars)}
        </text>
      ))}
      {brush === undefined ? null : (
        <rect x={x - 5} y={yOf(brush[1])} width={10} height={Math.max(2, yOf(brush[0]) - yOf(brush[1]))} fill="var(--accent)" opacity={0.35} rx={2} />
      )}
      {drag === null ? null : <rect x={x - 5} y={drag.y} width={10} height={drag.height} fill="var(--accent)" opacity={0.5} rx={2} />}
      <rect
        x={x - BRUSH_HIT_WIDTH / 2}
        y={top}
        width={BRUSH_HIT_WIDTH}
        height={height}
        fill="transparent"
        className="cursor-crosshair"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
    </g>
  );
}

function clip(label: string, maxChars: number): string {
  return label.length > maxChars ? `${label.slice(0, maxChars)}…` : label;
}

function rectOf(box: Box): { x: number; y: number; width: number; height: number } {
  return { x: box.x0, y: box.y0, width: box.x1 - box.x0, height: box.y1 - box.y0 };
}

function pointsOf(segment: [number, number][], axisX: (index: number) => number, yOf: (position: number) => number): string {
  return segment.map(([index, position]) => `${axisX(index)},${yOf(position)}`).join(" ");
}

/** Axis slot whose x is closest to the dragged label. */
function nearestIndex(x: number, count: number, axisX: (index: number) => number): number {
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < count; index += 1) {
    const distance = Math.abs(axisX(index) - x);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
}
