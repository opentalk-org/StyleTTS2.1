import { BarChart3, Columns3, GitBranch, Images } from "lucide-react";
import { lazy, Suspense, useMemo } from "react";

import { useViewerStore } from "@/features/viewer/store";
import { useVisibleUpdates } from "@/features/updates/query";
import type { ChartTheme } from "@/shared/chart";
import type { PanelTab, ProjectColumns, Run } from "@/shared/types";
import { ProgressLine, Tabs, type TabItem } from "@/shared/ui";

import { ComparePanel } from "@/features/compare/ComparePanel";

import { ChartsPanel } from "./ChartsPanel";
import { artifactNames } from "./logic";
import { MediaPanel } from "./MediaPanel";
import { useArtifactsQuery } from "./query";

const LineagePanel = lazy(() =>
  import("@/features/lineage/LineagePanel").then((module) => ({ default: module.LineagePanel })),
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
  onRevealRun: (runId: string, additive: boolean) => void;
}

export function Panels({ runs, scopedRuns, allRuns, projectColumns, runColors, chart, onRevealRun }: PanelsProps) {
  const tab = useViewerStore((state) => state.tab);
  const setTab = useViewerStore((state) => state.setTab);
  const projectId = useViewerStore((state) => state.projectId);
  const selectedRunIds = useViewerStore((state) => state.selectedRunIds);
  const runScope = useViewerStore((state) => state.runScope);
  const artifactsQuery = useArtifactsQuery(runs, tab === "media");
  // Charts read the scoped runs; the list of charts still comes from the selected ones, so
  // an ancestor with extra metrics does not add charts nobody asked for.
  const scopedRunIds = useMemo(() => scopedRuns.map((run) => run.id), [scopedRuns]);
  const plots = useMemo(
    () => [...new Set(runs.flatMap((run) => Object.keys(run.summary)))].sort()
      .map((name) => ({ name, series: [], pointCount: 0 })),
    [runs],
  );
  const artifacts = useMemo(
    () => (artifactsQuery.data ?? []).filter((artifact) => artifact.name !== "monitor/model_graph.json"),
    [artifactsQuery.data],
  );
  const mediaCount = artifactNames(artifacts).length;

  const items: TabItem<PanelTab>[] = [
    { id: "charts", label: "Charts", icon: <BarChart3 />, count: plots.length > 0 ? plots.length : undefined },
    { id: "media", label: "Media", icon: <Images />, count: mediaCount > 0 ? mediaCount : undefined },
    { id: "compare", label: "Compare", icon: <Columns3 /> },
    { id: "lineage", label: "Lineage", icon: <GitBranch /> },
  ];

  const activeTab = tab;
  const updateRunIds = activeTab === "lineage"
    ? allRuns.map((run) => run.id)
    : activeTab === "charts"
      ? scopedRunIds
      : selectedRunIds;
  const watchLineage = activeTab === "lineage" || (activeTab === "charts" && runScope === "lineage");
  useVisibleUpdates({ projectId, runIds: updateRunIds, tab: activeTab, watchLineage, enabled: true });

  return (
    <section aria-label="Panels" className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <ProgressLine active={artifactsQuery.isFetching} />
      <Tabs label="Panels" items={items} value={activeTab} onValue={setTab} />
      {activeTab === "charts" ? (
        <ChartsPanel
          runs={scopedRuns}
          runColors={runColors}
          chart={chart}
          plots={plots}
        />
      ) : null}
      {activeTab === "compare" ? <ComparePanel projectColumns={projectColumns} runs={scopedRuns} runColors={runColors} chart={chart} /> : null}
      {activeTab === "media" ? <MediaPanel runs={runs} runColors={runColors} artifacts={artifacts} /> : null}
      {activeTab === "lineage" ? (
        <Suspense fallback={null}>
          <LineagePanel allRuns={allRuns} runColors={runColors} chart={chart} onRevealRun={onRevealRun} />
        </Suspense>
      ) : null}
    </section>
  );
}
