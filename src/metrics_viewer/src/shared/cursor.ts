import type { Data, PlotHoverEvent } from "plotly.js";
import { useEffect, type RefObject } from "react";
import { create } from "zustand";

import { Plotly } from "@/shared/plot";
import type { XAxis } from "@/shared/types";

export interface CursorState {
  /** Hovered x in data units of `axis`, or null when nothing is hovered. */
  x: number | null;
  axis: XAxis;
  /** Chart the pointer is over; only that chart shows the hover box. */
  source: string | null;
  /** Run whose line is nearest the pointer; highlighted on every chart. */
  runId: string | null;
  /**
   * Runs highlighted together with it — the whole lineage the hovered run belongs to when
   * ancestors are drawn. Null highlights the hovered run alone.
   */
  runIds: ReadonlySet<string> | null;
  pointerX: number;
  pointerY: number;
  set: (next: {
    x: number | null;
    axis: XAxis;
    source: string;
    runId: string | null;
    runIds?: ReadonlySet<string> | null;
    pointerX: number;
    pointerY: number;
  }) => void;
  clear: () => void;
}

/**
 * Shared chart cursor. Kept outside React so hovering one chart moves a DOM overlay on
 * every other chart without re-rendering cards or calling Plotly.react.
 */
export const useCursorStore = create<CursorState>((set) => ({
  x: null,
  axis: "step",
  source: null,
  runId: null,
  runIds: null,
  pointerX: 0,
  pointerY: 0,
  set: (next) => set({ runIds: null, ...next }),
  clear: () => set({ x: null, source: null, runId: null, runIds: null }),
}));

interface PlotlyAxis {
  _offset: number;
  _length: number;
  d2p: (value: number) => number;
  /** Pixel to calcdata; milliseconds on a date axis, which is what the series hold. */
  p2c?: (pixel: number) => number;
  p2l?: (pixel: number) => number;
}

interface TraceMeta {
  runId: string;
  opacity: number;
  /** Line width the trace is drawn at when nothing is hovered. */
  width: number;
}

export interface PlotlyGraphDiv extends HTMLElement {
  _fullLayout: { xaxis: PlotlyAxis; yaxis: PlotlyAxis; _size: { t: number; h: number } };
  data: { meta?: TraceMeta; line?: { width?: number } }[];
}

/**
 * Data x exactly under the pointer. Plotly reports the sample it snapped to, which on a
 * sparse series sits visibly away from the cursor, so the shared line is placed from the
 * pointer position instead and only the values are read from the nearest sample.
 */
export function pointerDataX(event: PlotHoverEvent, graph: HTMLElement | null): number {
  const plotly = graph as PlotlyGraphDiv | null;
  const snapped = event.points[0].x as number;
  if (plotly === null || plotly._fullLayout === undefined) return snapped;
  const xaxis = plotly._fullLayout.xaxis;
  const toData = xaxis.p2c ?? xaxis.p2l;
  if (toData === undefined) return snapped;
  const pixel = event.event.clientX - plotly.getBoundingClientRect().left - xaxis._offset;
  const value = toData.call(xaxis, pixel);
  return Number.isFinite(value) ? value : snapped;
}

/** The run whose point at the hovered x is closest to the pointer, in pixels. */
export function nearestRun(event: PlotHoverEvent, graph: HTMLElement | null): string | null {
  const plotly = graph as PlotlyGraphDiv | null;
  if (plotly === null || plotly._fullLayout === undefined) return null;
  const yaxis = plotly._fullLayout.yaxis;
  const pointerY = event.event.clientY - plotly.getBoundingClientRect().top - yaxis._offset;
  let best: string | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const point of event.points) {
    const distance = Math.abs(yaxis.d2p(point.y as number) - pointerY);
    const meta = (point.data as { meta?: TraceMeta }).meta;
    if (meta !== undefined && distance < bestDistance) {
      bestDistance = distance;
      best = meta.runId;
    }
  }
  return best;
}

/** Positions a vertical overlay line at the shared cursor using Plotly's axis mapping. */
export function useCursorOverlay(
  graphRef: RefObject<HTMLElement | null>,
  overlayRef: RefObject<HTMLDivElement | null>,
  axis: XAxis,
) {
  useEffect(() => {
    function render(state: CursorState) {
      const overlay = overlayRef.current;
      const graph = graphRef.current as PlotlyGraphDiv | null;
      if (overlay === null) return;
      if (state.x === null || state.axis !== axis || graph === null || graph._fullLayout === undefined) {
        overlay.style.display = "none";
        return;
      }
      const xaxis = graph._fullLayout.xaxis;
      const pixel = xaxis.d2p(state.x);
      if (!Number.isFinite(pixel) || pixel < 0 || pixel > xaxis._length) {
        overlay.style.display = "none";
        return;
      }
      // Plotly's offsets are relative to the graph, the overlay to its positioned parent.
      // They differ wherever the plot sits inside padding, so measure instead of assuming.
      // Measure after unhiding: a display:none element reports no offsetParent at all.
      overlay.style.display = "block";
      const origin = overlay.offsetParent ?? overlay.parentElement ?? graph;
      const graphBox = graph.getBoundingClientRect();
      const originBox = origin.getBoundingClientRect();
      overlay.style.left = `${graphBox.left - originBox.left + xaxis._offset + pixel}px`;
      overlay.style.top = `${graphBox.top - originBox.top + graph._fullLayout._size.t}px`;
      overlay.style.height = `${graph._fullLayout._size.h}px`;
    }
    render(useCursorStore.getState());
    return useCursorStore.subscribe(render);
  }, [graphRef, overlayRef, axis]);
}

const DIMMED = 0.25;
/** The hovered lineage is lifted a little as well, so it reads as picked out, not just left behind. */
const LIT_OPACITY = 1.4;
const LIT_WIDTH = 0.75;
const HIGHLIGHT_DELAY = 60;

function inViewport(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  return rect.bottom > 0 && rect.top < window.innerHeight;
}

/** The runs a hover keeps at full opacity: the whole lineage when it has one. */
export function highlightedRuns(state: CursorState): ReadonlySet<string> | null {
  if (state.runId === null) return null;
  return state.runIds ?? new Set([state.runId]);
}

/**
 * Dims every trace outside the highlighted lineage. Restyles the graph directly, debounced
 * and only for charts on screen, so the pointer crossing lines does not trigger a redraw
 * storm.
 */
export function useRunHighlight(graphRef: RefObject<HTMLElement | null>) {
  useEffect(() => {
    let applied = "";
    let timer: number | null = null;
    function apply(runIds: ReadonlySet<string> | null) {
      const graph = graphRef.current as PlotlyGraphDiv | null;
      const key = runIds === null ? "" : [...runIds].sort().join(",");
      if (graph === null || graph.data === undefined || key === applied || !inViewport(graph)) return;
      applied = key;
      const lit = (meta: TraceMeta) => runIds !== null && runIds.has(meta.runId);
      const opacity = graph.data.map((trace) => {
        const meta = trace.meta;
        if (meta === undefined) return 1;
        if (runIds === null) return meta.opacity;
        return lit(meta) ? Math.min(1, meta.opacity * LIT_OPACITY) : meta.opacity * DIMMED;
      });
      const width = graph.data.map((trace) => {
        const meta = trace.meta;
        const base = meta?.width ?? trace.line?.width ?? 1.5;
        return meta !== undefined && lit(meta) ? base + LIT_WIDTH : base;
      });
      void Plotly.restyle(graph, { opacity, "line.width": width } as unknown as Data);
    }
    const unsubscribe = useCursorStore.subscribe((state) => {
      if (timer !== null) window.clearTimeout(timer);
      const runIds = highlightedRuns(state);
      timer = window.setTimeout(() => apply(runIds), HIGHLIGHT_DELAY);
    });
    return () => {
      unsubscribe();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [graphRef]);
}
