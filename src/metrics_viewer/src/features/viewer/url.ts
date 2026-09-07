import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { z } from "zod";

import { defaultRunColumns } from "@/shared/metrics";
import type { PanelTab } from "@/shared/types";

import type { ViewerSearch } from "./search";
import { DEFAULT_COMPARE, DEFAULT_GLOBAL_PLOT, DEFAULT_SQL, useViewerStore } from "./store";

const panelColumnsSchema = z.enum(["1", "2", "3", "auto"]);
const plotSettingsSchema = z.object({
  xAxis: z.enum(["inherit", "step", "lineage", "relative", "wall"]),
  xScale: z.enum(["linear", "log"]),
  yScale: z.enum(["linear", "log"]),
  smoothing: z.enum(["inherit", "none", "ema", "mean"]),
  smoothingValue: z.number(),
  renderMode: z.enum(["line", "scatter", "line-scatter"]),
});
const viewSchema = z.object({
  runScope: z.enum(["selected", "lineage"]),
  columns: z.array(z.string()),
  runColorOverrides: z.record(z.string(), z.string()),
  globalPlot: z.object({ xAxis: z.enum(["step", "lineage", "relative", "wall"]), smoothing: z.number() }),
  plotSettings: z.record(z.string(), plotSettingsSchema),
  hiddenPlots: z.array(z.string()),
  pinnedSections: z.array(z.string()),
  plotOrder: z.array(z.string()),
  compare: z.object({
    columns: z.array(z.string()).nullable(),
    plots: z.array(
      z.object({
        id: z.string(),
        // Plots saved before parallel coordinates existed carry neither kind nor axes.
        kind: z.enum(["chart", "parallel"]).default("chart"),
        x: z.string(),
        y: z.string().nullable(),
        logX: z.boolean(),
        logY: z.boolean(),
        dims: z.array(z.string()).nullable().default(null),
        colorBy: z.string().nullable().default(null),
      }),
    ),
  }),
  sql: z.string(),
  chartColumns: panelColumnsSchema,
  mediaColumns: panelColumnsSchema,
});

type ViewerState = ReturnType<typeof useViewerStore.getState>;
type UrlView = z.infer<typeof viewSchema>;

function viewOf(state: ViewerState): UrlView {
  return {
    runScope: state.runScope,
    columns: state.columns,
    runColorOverrides: state.runColorOverrides,
    globalPlot: state.globalPlot,
    plotSettings: state.plotSettings,
    hiddenPlots: state.hiddenPlots,
    pinnedSections: state.pinnedSections,
    plotOrder: state.plotOrder,
    compare: state.compare,
    sql: state.sql,
    chartColumns: state.chartColumns,
    mediaColumns: state.mediaColumns,
  };
}

function defaultView(): UrlView {
  return {
    runScope: "selected",
    columns: defaultRunColumns([]),
    runColorOverrides: {},
    globalPlot: DEFAULT_GLOBAL_PLOT,
    plotSettings: {},
    hiddenPlots: [],
    pinnedSections: [],
    plotOrder: [],
    compare: DEFAULT_COMPARE,
    sql: DEFAULT_SQL,
    chartColumns: "auto",
    mediaColumns: "auto",
  };
}

function parseView(value: string | undefined): UrlView {
  if (value === undefined) return defaultView();
  try {
    const parsed = viewSchema.safeParse(JSON.parse(value) as unknown);
    return parsed.success ? parsed.data : defaultView();
  } catch {
    return defaultView();
  }
}

function toSearch(state: ViewerState): ViewerSearch {
  return {
    project: state.projectId ?? undefined,
    runs: state.selectedRunIds.length === 0 ? undefined : state.selectedRunIds.join(","),
    tab: state.projectId === null || state.tab === "charts" ? undefined : state.tab,
    view: state.projectId === null ? undefined : JSON.stringify(viewOf(state)),
  };
}

function sameSearch(a: ViewerSearch, b: ViewerSearch): boolean {
  return a.project === b.project && a.runs === b.runs && a.tab === b.tab && a.view === b.view;
}

/** The URL is the portable representation of viewer state and drives history navigation. */
export function useUrlSync(search: ViewerSearch) {
  const navigate = useNavigate();

  useEffect(() => {
    const state = useViewerStore.getState();
    if (sameSearch(toSearch(state), search)) return;
    state.hydrate({
      projectId: search.project ?? null,
      selectedRunIds: search.runs === undefined ? [] : search.runs.split(","),
      tab: (search.tab ?? "charts") as PanelTab,
      view: parseView(search.view),
    });
    const normalized = toSearch(useViewerStore.getState());
    if (!sameSearch(normalized, search)) void navigate({ to: "/", search: normalized, replace: true });
  }, [navigate, search]);

  useEffect(
    () => useViewerStore.subscribe((state) => {
      const next = toSearch(state);
      if (sameSearch(next, search)) return;
      void navigate({ to: "/", search: next, replace: true });
    }),
    [navigate, search],
  );
}
