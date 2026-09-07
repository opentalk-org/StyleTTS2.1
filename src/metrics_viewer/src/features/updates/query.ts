import { useQuery, useQueryClient, type InfiniteData } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";

import type { ArrayMetricSeries, Artifact, PanelTab } from "@/shared/types";
import type { LogPage } from "@/features/logs/server";

import { initialUpdateCursor, pollVisibleChanges, type UpdateCursor } from "./server";

interface VisibleUpdates {
  projectId: string | null;
  runIds: string[];
  tab: PanelTab | "run-logs" | "run-graph";
  watchLineage: boolean;
  enabled: boolean;
}

export function useVisibleUpdates({ projectId, runIds, tab, watchLineage, enabled }: VisibleUpdates) {
  const queryClient = useQueryClient();
  const cursors = useRef(new Map<string, UpdateCursor>());
  const processedAt = useRef(new Map<string, number>());
  const runKey = [...runIds].sort().join(",");
  const ids = useMemo(() => runKey === "" ? [] : runKey.split(","), [runKey]);
  const scope = `${projectId ?? ""}:${tab}:${watchLineage}:${runKey}`;
  const updates = useQuery({
    queryKey: ["visible-updates", projectId, ids, tab, watchLineage],
    queryFn: () => pollVisibleChanges({ data: {
      projectId: projectId as string,
      runIds: ids,
      watchMetrics: tab === "charts" || tab === "compare" || tab === "lineage",
      watchArrayMetrics: tab === "run-graph",
      watchArtifacts: tab === "media" || tab === "run-graph",
      watchLineage,
      watchLogs: tab === "run-logs",
      cursor: cursors.current.get(scope) ?? initialUpdateCursor,
    } }),
    enabled: enabled && projectId !== null,
    refetchInterval: 3_000,
    refetchIntervalInBackground: true,
    staleTime: Infinity,
  });

  useEffect(() => {
    const change = updates.data;
    if (change === undefined || updates.dataUpdatedAt <= (processedAt.current.get(scope) ?? 0)) return;
    processedAt.current.set(scope, updates.dataUpdatedAt);
    cursors.current.set(scope, change.cursor);

    if (change.baseline.status || change.runs.length > 0) {
      void queryClient.invalidateQueries({ queryKey: ["project-bootstrap", projectId] });
    }
    applyMetricChanges(queryClient, projectId, change, ids);
    applyArrayMetricChanges(queryClient, change, ids);
    applyArtifactChanges(queryClient, change, ids);
    applyLogChanges(queryClient, change, ids);
    if (change.baseline.lineage || change.lineageChanged
      || (watchLineage && (change.runs.length > 0 || change.metrics.length > 0))) {
      void queryClient.invalidateQueries({ queryKey: ["lineage", projectId] });
    }
  }, [updates.data, updates.dataUpdatedAt, queryClient, projectId, ids, scope, watchLineage]);

  return updates;
}

type QueryClient = ReturnType<typeof useQueryClient>;
type VisibleChange = Awaited<ReturnType<typeof pollVisibleChanges>>;

function applyMetricChanges(
  queryClient: QueryClient,
  projectId: string | null,
  change: VisibleChange,
  runIds: string[],
) {
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
  }
  if (!change.baseline.metrics && change.metrics.length === 0) return;
  void queryClient.invalidateQueries({ queryKey: ["run-details"] });
  void queryClient.invalidateQueries({ queryKey: ["run-metrics", projectId] });
  void queryClient.invalidateQueries({ queryKey: ["project-bootstrap", projectId] });
  void queryClient.invalidateQueries({ queryKey: ["plots"] });
  if (change.baseline.metrics) {
    void queryClient.invalidateQueries({
      predicate: (candidate) => candidate.queryKey[0] === "plot-range"
        && Array.isArray(candidate.queryKey[1])
        && candidate.queryKey[1].some((runId) => runIds.includes(String(runId))),
    });
  }
}

function applyArrayMetricChanges(queryClient: QueryClient, change: VisibleChange, runIds: string[]) {
  if (change.baseline.arrayMetrics) {
    for (const runId of runIds) {
      void queryClient.invalidateQueries({ queryKey: ["array-metric-names", runId] });
      void queryClient.invalidateQueries({ queryKey: ["array-metric", runId] });
    }
  }
  for (const runId of runIds) {
    const changedNames = change.arrayMetrics
      .filter((metric) => metric.runId === runId)
      .map((metric) => metric.name);
    if (changedNames.length === 0) continue;
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

function applyArtifactChanges(queryClient: QueryClient, change: VisibleChange, runIds: string[]) {
  if (change.baseline.artifacts) {
    void queryClient.invalidateQueries({ queryKey: ["artifacts", runIds] });
    void queryClient.invalidateQueries({ queryKey: ["model-graph", runIds[0]] });
  }
  if (change.artifacts.length === 0) return;
  queryClient.setQueryData<Artifact[]>(["artifacts", runIds], (current) => {
    const merged = new Map((current ?? []).map((artifact) => [artifact.id, artifact]));
    for (const artifact of change.artifacts) merged.set(artifact.id, artifact);
    return [...merged.values()].sort((a, b) => a.name.localeCompare(b.name) || a.step - b.step);
  });
  if (change.artifacts.some((artifact) => artifact.name === "monitor/model_graph.json")) {
    void queryClient.invalidateQueries({ queryKey: ["model-graph", runIds[0]] });
  }
}

function applyLogChanges(queryClient: QueryClient, change: VisibleChange, runIds: string[]) {
  if (change.baseline.logs) {
    void queryClient.invalidateQueries({ queryKey: ["logs", runIds] });
  }
  if (change.logs.length === 0) return;
  queryClient.setQueryData<InfiniteData<LogPage>>(["logs", runIds], (current) => {
    if (current === undefined) return current;
    const incoming = change.logs.map((row) => ({
      runId: row.runId,
      timestamp: Number(row.timestampMs),
      message: row.message,
    }));
    const keys = new Set(incoming.map(logKey));
    const pages = current.pages.map((page) => ({
      ...page,
      rows: page.rows.filter((row) => !keys.has(logKey(row))),
    }));
    pages[0] = { ...pages[0], rows: [...incoming, ...pages[0].rows] };
    return { ...current, pages };
  });
}

function logKey(log: { runId: string; timestamp: number; message: string }) {
  return `${log.runId}:${log.timestamp}:${log.message}`;
}
