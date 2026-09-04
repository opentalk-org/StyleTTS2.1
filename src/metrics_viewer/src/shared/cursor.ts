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
  pointerY: number;
  set: (next: { x: number | null; axis: XAxis; source: string; runId: string | null; pointerY: number }) => void;
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
  pointerY: 0,
  set: (next) => set(next),
  clear: () => set({ x: null, source: null, runId: null }),
}));

interface PlotlyAxis {
  _offset: number;
  _length: number;
  d2p: (value: number) => number;
}

interface TraceMeta {
  runId: string;
  opacity: number;
}

export interface PlotlyGraphDiv extends HTMLElement {
  _fullLayout: { xaxis: PlotlyAxis; yaxis: PlotlyAxis; _size: { t: number; h: number } };
  data: { meta?: TraceMeta }[];
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
      overlay.style.display = "block";
      overlay.style.left = `${xaxis._offset + pixel}px`;
      overlay.style.top = `${graph._fullLayout._size.t}px`;
      overlay.style.height = `${graph._fullLayout._size.h}px`;
    }
    render(useCursorStore.getState());
    return useCursorStore.subscribe(render);
  }, [graphRef, overlayRef, axis]);
}

const DIMMED = 0.25;
const HIGHLIGHT_DELAY = 60;

function inViewport(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  return rect.bottom > 0 && rect.top < window.innerHeight;
}

/**
 * Dims every trace except the highlighted run. Restyles the graph directly, debounced and
 * only for charts on screen, so the pointer crossing lines does not trigger a redraw storm.
 */
export function useRunHighlight(graphRef: RefObject<HTMLElement | null>) {
  useEffect(() => {
    let applied: string | null = null;
    let timer: number | null = null;
    function apply(runId: string | null) {
      const graph = graphRef.current as PlotlyGraphDiv | null;
      if (graph === null || graph.data === undefined || runId === applied || !inViewport(graph)) return;
      applied = runId;
      const opacity = graph.data.map((trace) => {
        const meta = trace.meta;
        if (meta === undefined) return 1;
        return runId === null || meta.runId === runId ? meta.opacity : meta.opacity * DIMMED;
      });
      void Plotly.restyle(graph, { opacity } as unknown as Data);
    }
    const unsubscribe = useCursorStore.subscribe((state) => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => apply(state.runId), HIGHLIGHT_DELAY);
    });
    return () => {
      unsubscribe();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [graphRef]);
}
