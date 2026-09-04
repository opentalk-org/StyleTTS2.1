import { useLayoutEffect, useMemo } from "react";

import { readChartTheme, type ChartTheme } from "@/shared/chart";

import { useViewerStore } from "./store";

/** Applies the stored theme to <html> and exposes the matching chart colours. */
export function useTheme(): { theme: "dark" | "light"; chart: ChartTheme; toggle: () => void } {
  const theme = useViewerStore((state) => state.theme);
  const setTheme = useViewerStore((state) => state.setTheme);

  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Read after the attribute is applied so the tokens resolve to the right palette.
  const chart = useMemo(() => {
    document.documentElement.dataset.theme = theme;
    return readChartTheme();
  }, [theme]);

  return { theme, chart, toggle: () => setTheme(theme === "dark" ? "light" : "dark") };
}
