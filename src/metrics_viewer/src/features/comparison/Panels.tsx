import { BarChart3, Columns3, Images, Network } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { lazy, Suspense, useMemo } from "react";

import { isDefaultSql, useViewerStore } from "@/features/viewer/store";
import { useVisibleUpdates } from "@/features/updates/query";
import type { ChartTheme } from "@/shared/chart";
import type { PanelTab, Run } from "@/shared/types";
import { ProgressLine, Tabs, type TabItem } from "@/shared/ui";

import { ComparePanel } from "@/features/compare/ComparePanel";

import { ChartsPanel } from "./ChartsPanel";
import { artifactNames, groupPlots } from "./logic";
import { MediaPanel } from "./MediaPanel";
import { useArtifactsQuery, useMetricNamesQuery, usePlotsQuery } from "./query";

const ModelMonitor = lazy(() =>
  import("@/features/model-monitor/ModelMonitor").then((module) => ({ default: module.ModelMonitor })),
);

interface PanelsProps {
  /** Selected runs, drawn in charts and media. */
  runs: Run[];
  /** Every run of the project, for the compare picker. */
  allRuns: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
}

export function Panels({ runs, allRuns, runColors, chart }: PanelsProps) {
  const queryClient = useQueryClient();
  const tab = useViewerStore((state) => state.tab);
  const setTab = useViewerStore((state) => state.setTab);
  const projectId = useViewerStore((state) => state.projectId);
  const runningSql = useViewerStore((state) => state.runningSql);
  const selectedRunIds = useViewerStore((state) => state.selectedRunIds);
  const defaultQuery = isDefaultSql(runningSql);
  const metricNamesQuery = useMetricNamesQuery(selectedRunIds, defaultQuery);
  const artifactsQuery = useArtifactsQuery(runs, tab === "media");
  const plotsQuery = usePlotsQuery(projectId, runningSql, selectedRunIds, !defaultQuery);

  const plots = useMemo(
    () => defaultQuery
      ? (metricNamesQuery.data ?? []).map((name) => ({ name, series: [], pointCount: 0 }))
      : groupPlots(plotsQuery.data ?? null),
    [defaultQuery, metricNamesQuery.data, plotsQuery.data],
  );
  const artifacts = useMemo(
    () => (artifactsQuery.data ?? []).filter((artifact) => artifact.name !== "monitor/model_graph.json"),
    [artifactsQuery.data],
  );
  const mediaCount = artifactNames(artifacts).length;
  useVisibleUpdates({ projectId, runIds: selectedRunIds, tab });

  const items: TabItem<PanelTab>[] = [
    { id: "charts", label: "Charts", icon: <BarChart3 />, count: plots.length > 0 ? plots.length : undefined },
    { id: "compare", label: "Compare", icon: <Columns3 /> },
    { id: "media", label: "Media", icon: <Images />, count: mediaCount > 0 ? mediaCount : undefined },
    {
      id: "graph",
      label: "Model graph",
      icon: <Network />,
      disabledReason: runs.length === 1 ? undefined : "Select exactly one run",
    },
  ];

  const activeTab: PanelTab = tab === "graph" && runs.length !== 1 ? "charts" : tab;

  return (
    <section aria-label="Panels" className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <ProgressLine active={plotsQuery.isFetching || artifactsQuery.isFetching} />
      <Tabs label="Panels" items={items} value={activeTab} onValue={setTab} />
      {activeTab === "charts" ? (
        <ChartsPanel
          runs={runs}
          runColors={runColors}
          chart={chart}
          plots={plots}
          result={plotsQuery.data ?? null}
          error={plotsQuery.error}
          fetching={plotsQuery.isFetching}
          onRefetch={() => {
            if (defaultQuery) void queryClient.invalidateQueries({ queryKey: ["plot-range"] });
            else void plotsQuery.refetch();
          }}
          rangeQueries={defaultQuery}
        />
      ) : null}
      {activeTab === "compare" ? <ComparePanel allRuns={allRuns} runs={runs} runColors={runColors} chart={chart} /> : null}
      {activeTab === "media" ? <MediaPanel runs={runs} runColors={runColors} artifacts={artifacts} /> : null}
      {activeTab === "graph" ? (
        <Suspense fallback={null}>
          <ModelMonitor run={runs[0]} chart={chart} />
        </Suspense>
      ) : null}
    </section>
  );
}
