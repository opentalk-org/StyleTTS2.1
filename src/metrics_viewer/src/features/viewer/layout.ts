import { useEffect, useState } from "react";

import type { SplitCollapsed, SplitOrientation } from "@/shared/ui";

const LAYOUT_KEY = "runflow.metrics.layout.v2";
const STACK_BELOW = 1100;

export interface ViewerLayout {
  orientation: SplitOrientation;
  /** Runs pane size in px along the split axis. */
  size: number;
  collapsed: SplitCollapsed;
}

export const DEFAULT_LAYOUT: ViewerLayout = { orientation: "columns", size: 440, collapsed: null };

export function useViewerLayout() {
  const [layout, setLayout] = useState<ViewerLayout>(loadLayout);
  const narrow = useNarrowViewport();

  useEffect(() => {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  }, [layout]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    return () => cancelAnimationFrame(frame);
  }, [layout.orientation, layout.collapsed, narrow]);

  function patchLayout(patch: Partial<ViewerLayout>) {
    setLayout((current) => ({ ...current, ...patch }));
  }

  function toggleCollapsed(pane: "start" | "end") {
    patchLayout({ collapsed: layout.collapsed === pane ? null : pane });
  }

  // Narrow viewports always stack; the stored orientation is kept for when the window grows.
  const effective: ViewerLayout = narrow
    ? { ...layout, orientation: "rows", size: Math.min(layout.size, 320) }
    : layout;

  return { layout: effective, patchLayout, toggleCollapsed, resetLayout: () => setLayout(DEFAULT_LAYOUT) };
}

function useNarrowViewport(): boolean {
  const [narrow, setNarrow] = useState(() => window.innerWidth < STACK_BELOW);
  useEffect(() => {
    const media = window.matchMedia(`(max-width: ${STACK_BELOW - 1}px)`);
    const onChange = () => setNarrow(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

function loadLayout(): ViewerLayout {
  try {
    const stored = localStorage.getItem(LAYOUT_KEY);
    return stored === null ? DEFAULT_LAYOUT : { ...DEFAULT_LAYOUT, ...(JSON.parse(stored) as ViewerLayout) };
  } catch {
    return DEFAULT_LAYOUT;
  }
}
