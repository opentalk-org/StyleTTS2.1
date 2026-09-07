import { useEffect, type RefObject } from "react";

import { Plotly } from "@/shared/plot";

/** Plotly listens to window resizing, but grid column changes only resize its container. */
export function usePlotResize(containerRef: RefObject<HTMLElement | null>, graphRef: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const container = containerRef.current;
    if (container === null) return;
    let frame: number | null = null;
    const observer = new ResizeObserver(() => {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (graphRef.current !== null) void Plotly.Plots.resize(graphRef.current);
      });
    });
    observer.observe(container);
    return () => {
      observer.disconnect();
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [containerRef, graphRef]);
}
