import { useEffect, useState } from "react";

import type { SplitCollapsed, SplitOrientation } from "@/shared/ui";

const LAYOUT_KEY = "runflow.metrics.layout.v1";

export interface ViewerLayout {
  orientation: SplitOrientation;
  ratio: number;
  collapsed: SplitCollapsed;
}

const DEFAULT_LAYOUT: ViewerLayout = { orientation: "columns", ratio: 0.38, collapsed: null };

export function useViewerLayout() {
  const [layout, setLayout] = useState<ViewerLayout>(loadLayout);

  useEffect(() => {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  }, [layout]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    return () => cancelAnimationFrame(frame);
  }, [layout.orientation, layout.collapsed]);

  function patchLayout(patch: Partial<ViewerLayout>) {
    setLayout((current) => ({ ...current, ...patch }));
  }

  function toggleCollapsed(pane: "start" | "end") {
    patchLayout({ collapsed: layout.collapsed === pane ? null : pane });
  }

  return { layout, patchLayout, toggleCollapsed };
}

function loadLayout(): ViewerLayout {
  try {
    const stored = localStorage.getItem(LAYOUT_KEY);
    return stored === null
      ? DEFAULT_LAYOUT
      : { ...DEFAULT_LAYOUT, ...(JSON.parse(stored) as ViewerLayout) };
  } catch {
    return DEFAULT_LAYOUT;
  }
}
