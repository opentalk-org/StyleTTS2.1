import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";

import type { ArrayMetricSeries, Artifact, PanelTab, Run } from "@/shared/types";

import { initialUpdateCursor, pollVisibleChanges, type UpdateCursor } from "./server";

interface VisibleUpdates {
  projectId: string | null;
  runIds: string[];
  tab: PanelTab;
}

export function useVisibleUpdates({ projectId, runIds, tab }: VisibleUpdates) {
  const queryClient = useQueryClient();
  const cursor = useRef<UpdateCursor>(initialUpdateCursor);
  const cursorProject = useRef<string | null>(projectId);
  if (cursorProject.current !== projectId) {
    cursorProject.current = projectId;
    cursor.current = initialUpdateCursor;
  }
  const ids = useMemo(() => [...runIds].sort(), [runIds]);
  const updates = useQuery({
    queryKey: ["visible-updates", projectId, ids, tab],
    queryFn: () => pollVisibleChanges({ data: {
      projectId: projectId as string,
      runIds: ids,
      watchMetrics: tab === "charts" || tab === "compare",
      watchArrayMetrics: tab === "graph",
      watchArtifacts: tab === "media" || tab === "graph",
      cursor: cursor.current,
    } }),
    enabled: projectId !== null,
    refetchInterval: 3_000,
    refetchIntervalInBackground: false,
    staleTime: Infinity,
  });

  useEffect(() => {
    const change = updates.data;
    if (change === undefined) return;
    cursor.current = change.cursor;
    if (change.runs.length > 0) {
      queryClient.setQueryData<Run[]>(["runs", projectId], (current) => {
        const changed = new Map(change.runs.map((run) => [run.id, run]));
        const merged = (current ?? []).map((run) => changed.get(run.id) ?? run);
        const known = new Set(merged.map((run) => run.id));
        return [...change.runs.filter((run) => !known.has(run.id)), ...merged];
      });
    }
    for (const metric of change.metrics) {
      void queryClient.invalidateQueries({
        predicate: (candidate) => {
          const [kind, queryRunIds, name, xMin, xMax] = candidate.queryKey;
          if (kind !== "plot-range" || name !== metric.name || !Array.isArray(queryRunIds)) return false;
          if (!queryRunIds.includes(metric.runId)) return false;
          if (xMin === null || xMax === null) return true;
          return metric.maxStep >= Number(xMin) && metric.minStep <= Number(xMax);
        },
      });
      queryClient.setQueryData<Record<string, number>>(["run-summary", metric.runId], (current) => ({
        ...current,
        [metric.name]: Number(metric.value),
      }));
    }
    if (change.metrics.length > 0) {
      queryClient.setQueryData<string[]>(["metric-names", ids], (current) => [
        ...new Set([...(current ?? []), ...change.metrics.map((metric) => metric.name)]),
      ].sort());
      void queryClient.invalidateQueries({ queryKey: ["plots"] });
    }
    if (change.arrayMetrics.length > 0) {
      for (const runId of ids) {
        const changedNames = change.arrayMetrics
          .filter((metric) => metric.runId === runId)
          .map((metric) => metric.name);
        queryClient.setQueryData<string[]>(["array-metric-names", runId], (current) => [
          ...new Set([...(current ?? []), ...changedNames]),
        ].sort());
      }
      for (const metric of change.arrayMetrics) {
        queryClient.setQueryData<ArrayMetricSeries>(["array-metric", metric.runId, metric.name], (current) => {
          const byStep = new Map((current?.steps ?? []).map((step, index) => [step, {
            timestamp: current?.timestamps[index] as number,
            value: current?.values[index] as number[],
          }]));
          byStep.set(Number(metric.step), {
            timestamp: Number(metric.timestampMs),
            value: metric.value.map(Number),
          });
          const ordered = [...byStep.entries()].sort((a, b) => a[0] - b[0]);
          return {
            name: metric.name,
            steps: ordered.map(([step]) => step),
            timestamps: ordered.map(([, value]) => value.timestamp),
            values: ordered.map(([, value]) => value.value),
          };
        });
      }
    }
    if (change.artifacts.length > 0) {
      queryClient.setQueriesData<Artifact[]>({ queryKey: ["artifacts"] }, (current) => {
        const merged = new Map((current ?? []).map((artifact) => [artifact.id, artifact]));
        for (const artifact of change.artifacts) merged.set(artifact.id, artifact);
        return [...merged.values()].sort((a, b) => a.name.localeCompare(b.name) || a.step - b.step);
      });
      void queryClient.invalidateQueries({ queryKey: ["model-graph"] });
    }
  }, [updates.data, queryClient, projectId, ids]);

  return updates;
}
