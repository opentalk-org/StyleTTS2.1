import { BarChart3, Columns3, GitBranch, Images, Network } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { lazy, Suspense, useMemo } from "react";

import { useRunLineage } from "@/features/lineage/query";
import { isDefaultSql, useViewerStore } from "@/features/viewer/store";
import { useVisibleUpdates } from "@/features/updates/query";
import type { ChartTheme } from "@/shared/chart";
import type { PanelTab, ProjectColumns, Run } from "@/shared/types";
import { ProgressLine, Tabs, type TabItem } from "@/shared/ui";

import { ComparePanel } from "@/features/compare/ComparePanel";

import { ChartsPanel } from "./ChartsPanel";
import { artifactNames, groupPlots } from "./logic";
import { MediaPanel } from "./MediaPanel";
import { useArtifactsQuery, usePlotsQuery } from "./query";

const LineagePanel = lazy(() =>
  import("@/features/lineage/LineagePanel").then((module) => ({ default: module.LineagePanel })),
);

const ModelMonitor = lazy(() =>
  import("@/features/model-monitor/ModelMonitor").then((module) => ({ default: module.ModelMonitor })),
);

interface PanelsProps {
  /** Selected runs, drawn in media and used for the list of charts. */
  runs: Run[];
  /** Selected runs plus their ancestors when the lineage scope is on; what charts draw. */
  scopedRuns: Run[];
  /** Every run of the project, for the compare picker. */
  allRuns: Run[];
  projectColumns: ProjectColumns;
  runColors: Record<string, string>;
  chart: ChartTheme;
}

export function Panels({ runs, scopedRuns, allRuns, projectColumns, runColors, chart }: PanelsProps) {
  const queryClient = useQueryClient();
  const lineage = useRunLineage();
  const tab = useViewerStore((state) => state.tab);
  const setTab = useViewerStore((state) => state.setTab);
  const projectId = useViewerStore((state) => state.projectId);
  const runningSql = useViewerStore((state) => state.runningSql);
  const selectedRunIds = useViewerStore((state) => state.selectedRunIds);
  const defaultQuery = isDefaultSql(runningSql);
  const artifactsQuery = useArtifactsQuery(runs, tab === "media");
  // Charts read the scoped runs; the list of charts still comes from the selected ones, so
  // an ancestor with extra metrics does not add charts nobody asked for.
  const scopedRunIds = useMemo(() => scopedRuns.map((run) => run.id), [scopedRuns]);
  const plotsQuery = usePlotsQuery(projectId, runningSql, scopedRunIds, !defaultQuery);

  const plots = useMemo(
    () => defaultQuery
      ? [...new Set(runs.flatMap((run) => Object.keys(run.summary)))].sort()
          .map((name) => ({ name, series: [], pointCount: 0 }))
      : groupPlots(plotsQuery.data ?? null, lineage.offsets),
    [defaultQuery, lineage.offsets, plotsQuery.data, runs],
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
    { id: "lineage", label: "Lineage", icon: <GitBranch /> },
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
          runs={scopedRuns}
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
      {activeTab === "compare" ? <ComparePanel projectColumns={projectColumns} runs={scopedRuns} runColors={runColors} chart={chart} /> : null}
      {activeTab === "media" ? <MediaPanel runs={runs} runColors={runColors} artifacts={artifacts} /> : null}
      {activeTab === "lineage" ? (
        <Suspense fallback={null}>
          <LineagePanel allRuns={allRuns} runColors={runColors} chart={chart} />
        </Suspense>
      ) : null}
      {activeTab === "graph" ? (
        <Suspense fallback={null}>
          <ModelMonitor run={runs[0]} chart={chart} />
        </Suspense>
      ) : null}
    </section>
  );
}
