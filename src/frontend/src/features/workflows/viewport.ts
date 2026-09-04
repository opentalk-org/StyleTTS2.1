import type { Viewport } from "./types";

export function graphPoint(viewport: Viewport, clientX: number, clientY: number, left: number, top: number) {
  return {
    x: (clientX - left - viewport.x) / viewport.zoom,
    y: (clientY - top - viewport.y) / viewport.zoom,
  };
}

export function zoomViewport(viewport: Viewport, nextZoom: number, anchorX: number, anchorY: number): Viewport {
  const zoom = Math.max(0.25, Math.min(2, nextZoom));
  const graphX = (anchorX - viewport.x) / viewport.zoom;
  const graphY = (anchorY - viewport.y) / viewport.zoom;
  return { x: anchorX - graphX * zoom, y: anchorY - graphY * zoom, zoom };
}
