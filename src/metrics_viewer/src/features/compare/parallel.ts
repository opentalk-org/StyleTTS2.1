import { formatDuration, formatRelative } from "@/features/runs/logic";
import { runColumnLabel } from "@/shared/metrics";
import type { Run, Scalar } from "@/shared/types";

import { rawValue } from "./logic";

/** A numeric axis switches to log when its values span at least this many decades. */
const LOG_DECADES = 3;
const NUMERIC_TICKS = 5;
const MAX_CATEGORY_TICKS = 10;

export interface AxisTick {
  /** 0 at the bottom of the axis, 1 at the top. */
  position: number;
  label: string;
}

export interface ParallelAxis {
  column: string;
  label: string;
  numeric: boolean;
  /** Numeric axis drawn in log10; `min`/`max` are then log10 of the data range. */
  log: boolean;
  min: number;
  max: number;
  /** Categorical axis values, bottom first; empty on a numeric axis. */
  categories: string[];
  ticks: AxisTick[];
}

export function buildAxes(runs: Run[], columns: string[]): ParallelAxis[] {
  return columns.map((column) => buildAxis(runs, column));
}

function buildAxis(runs: Run[], column: string): ParallelAxis {
  const values = runs.map((run) => rawValue(run, column)).filter((value): value is Scalar => value !== undefined);
  const numbers = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const base = { column, label: runColumnLabel(column) };
  if (numbers.length > 0 && numbers.length === values.length) {
    const low = Math.min(...numbers);
    const high = Math.max(...numbers);
    const log = low > 0 && high / low >= 10 ** LOG_DECADES;
    const min = log ? Math.log10(low) : low;
    const max = log ? Math.log10(high) : high;
    return { ...base, numeric: true, log, min, max, categories: [], ticks: numericTicks(column, min, max, log) };
  }
  const categories = [...new Set(values.map(String))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return { ...base, numeric: false, log: false, min: 0, max: 0, categories, ticks: categoryTicks(categories) };
}

function numericTicks(column: string, min: number, max: number, log: boolean): AxisTick[] {
  if (min === max) return [{ position: 0.5, label: tickLabel(column, log ? 10 ** min : min) }];
  return Array.from({ length: NUMERIC_TICKS }, (_, index) => {
    const position = index / (NUMERIC_TICKS - 1);
    const value = min + (max - min) * position;
    return { position, label: tickLabel(column, log ? 10 ** value : value) };
  });
}

/** Long category lists get evenly spaced labels so the axis stays readable. */
function categoryTicks(categories: string[]): AxisTick[] {
  const stride = Math.ceil(categories.length / MAX_CATEGORY_TICKS);
  return categories.flatMap((label, index) =>
    index % stride === 0 ? [{ position: categoryPosition(index, categories.length), label }] : [],
  );
}

function categoryPosition(index: number, count: number): number {
  return count <= 1 ? 0.5 : index / (count - 1);
}

function tickLabel(column: string, value: number): string {
  if (column === "duration") return formatDuration(value);
  if (column === "startedAt") return formatRelative(value);
  return compactNumber(value);
}

export function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e5 || magnitude < 1e-3)) return value.toExponential(1);
  return String(Number(value.toPrecision(4)));
}

/** Where a run sits on an axis, 0 at the bottom and 1 at the top; null when it has no value. */
export function axisPosition(axis: ParallelAxis, run: Run): number | null {
  const value = rawValue(run, axis.column);
  if (value === undefined) return null;
  if (!axis.numeric) {
    const index = axis.categories.indexOf(String(value));
    return index === -1 ? null : categoryPosition(index, axis.categories.length);
  }
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (axis.log && value <= 0) return null;
  const scaled = axis.log ? Math.log10(value) : value;
  if (axis.max === axis.min) return 0.5;
  return (scaled - axis.min) / (axis.max - axis.min);
}

export interface RunLine {
  run: Run;
  /** One entry per axis; null where the run has no value there, which breaks the line. */
  positions: (number | null)[];
}

/** Run positions per axis. A run with fewer than two values draws no segment and is dropped. */
export function runPositions(runs: Run[], axes: ParallelAxis[]): RunLine[] {
  return runs.flatMap((run) => {
    const positions = axes.map((axis) => axisPosition(axis, run));
    if (positions.filter((position) => position !== null).length < 2) return [];
    return [{ run, positions }];
  });
}

/** Runs of consecutive axes the line actually crosses, as [axis index, position] pairs. */
export function segments(positions: (number | null)[]): [number, number][][] {
  const result: [number, number][][] = [];
  let current: [number, number][] = [];
  positions.forEach((position, index) => {
    if (position === null) {
      if (current.length > 1) result.push(current);
      current = [];
      return;
    }
    current.push([index, position]);
  });
  if (current.length > 1) result.push(current);
  return result;
}

/** Normalized [low, high] selections keyed by column; a run must fall inside every one. */
export type Brushes = Record<string, [number, number]>;

/** A run with no value on a brushed axis cannot satisfy that brush, so it is filtered out. */
export function passesBrushes(axes: ParallelAxis[], positions: (number | null)[], brushes: Brushes): boolean {
  return axes.every((axis, index) => {
    const brush = brushes[axis.column];
    const position = positions[index];
    return brush === undefined || (position !== null && position >= brush[0] && position <= brush[1]);
  });
}

const GRADIENT = ["#3b6fd4", "#59c2d8", "#4fc38a", "#d9c04a", "#f0716c"];

/** Colour for a run when the plot is coloured by a metric; `t` runs 0..1 over its range. */
export function gradientColor(t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (GRADIENT.length - 1);
  const index = Math.min(GRADIENT.length - 2, Math.floor(scaled));
  return mix(GRADIENT[index], GRADIENT[index + 1], scaled - index);
}

function mix(from: string, to: string, amount: number): string {
  const a = Number.parseInt(from.slice(1), 16);
  const b = Number.parseInt(to.slice(1), 16);
  const channel = (shift: number) =>
    Math.round((((a >> shift) & 255) * (1 - amount)) + (((b >> shift) & 255) * amount));
  return `rgb(${channel(16)}, ${channel(8)}, ${channel(0)})`;
}

/** Colour per run: the run's own colour, or the gradient over a chosen numeric column. */
export function lineColors(
  runs: Run[],
  colorBy: string | null,
  runColors: Record<string, string>,
): Record<string, string> {
  if (colorBy === null) return runColors;
  const values = new Map<string, number>();
  for (const run of runs) {
    const value = rawValue(run, colorBy);
    if (typeof value === "number" && Number.isFinite(value)) values.set(run.id, value);
  }
  const numbers = [...values.values()];
  if (numbers.length === 0) return runColors;
  const low = Math.min(...numbers);
  const high = Math.max(...numbers);
  const colors: Record<string, string> = {};
  for (const run of runs) {
    const value = values.get(run.id);
    colors[run.id] = value === undefined ? runColors[run.id] : gradientColor(high === low ? 0.5 : (value - low) / (high - low));
  }
  return colors;
}

export function colorScaleRange(runs: Run[], colorBy: string): { low: number; high: number } | null {
  const numbers = runs
    .map((run) => rawValue(run, colorBy))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (numbers.length === 0) return null;
  return { low: Math.min(...numbers), high: Math.max(...numbers) };
}

export const GRADIENT_CSS = `linear-gradient(to right, ${GRADIENT.join(", ")})`;

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function normalizeBox(box: Box): Box {
  return {
    x0: Math.min(box.x0, box.x1),
    y0: Math.min(box.y0, box.y1),
    x1: Math.max(box.x0, box.x1),
    y1: Math.max(box.y0, box.y1),
  };
}

/** True when any part of the drawn line falls inside the selection rectangle. */
export function polylineHitsBox(points: [number, number][], box: Box): boolean {
  const area = normalizeBox(box);
  return points.some(
    ([x, y], index) =>
      inside(x, y, area) || (index > 0 && segmentHitsBox(points[index - 1], points[index], area)),
  );
}

function inside(x: number, y: number, box: Box): boolean {
  return x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1;
}

/** Liang–Barsky: the segment survives clipping against the rectangle, so the two overlap. */
function segmentHitsBox(from: [number, number], to: [number, number], box: Box): boolean {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  let enter = 0;
  let exit = 1;
  const edges: [number, number][] = [
    [-dx, from[0] - box.x0],
    [dx, box.x1 - from[0]],
    [-dy, from[1] - box.y0],
    [dy, box.y1 - from[1]],
  ];
  for (const [p, q] of edges) {
    if (p === 0) {
      if (q < 0) return false;
      continue;
    }
    const t = q / p;
    if (p < 0) enter = Math.max(enter, t);
    else exit = Math.min(exit, t);
    if (enter > exit) return false;
  }
  return true;
}
