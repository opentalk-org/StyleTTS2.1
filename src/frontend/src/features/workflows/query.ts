import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { fetchRunGraph, fetchRuns, fetchRunSnapshot, fetchSavedWorkflows, fetchWorkflowSchema, loadRunNode, saveWorkflow, startGraph, stopRun, unloadRunNode } from "./api";

const STATUS_STALE_MS = 10_000;

export function useWorkflowSchemaQuery() {
  return useQuery({ queryKey: ["workflow-schema"], queryFn: fetchWorkflowSchema });
}

export function useRunsQuery() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: fetchRuns,
    staleTime: STATUS_STALE_MS,
    refetchOnWindowFocus: false,
  });
}

export function useSavedWorkflowsQuery() {
  return useQuery({ queryKey: ["workflows"], queryFn: fetchSavedWorkflows });
}

export function useSaveWorkflowMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: saveWorkflow,
    onSuccess: (workflow) => {
      showToast(`Saved "${workflow.name}"`);
      queryClient.invalidateQueries({ queryKey: ["workflows"] });
    },
  });
}

export function useRunSnapshotQuery(runId: string | null) {
  return useQuery({
    queryKey: ["run-snapshot", runId],
    queryFn: () => fetchRunSnapshot(runId as string),
    enabled: runId !== null,
    staleTime: STATUS_STALE_MS,
    refetchOnWindowFocus: false,
  });
}

export function useRunGraphQuery(runId: string | null) {
  return useQuery({
    queryKey: ["run-graph", runId],
    queryFn: () => fetchRunGraph(runId as string),
    enabled: runId !== null,
  });
}

export function useStartGraphMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: startGraph,
    onSuccess: (run) => {
      showToast(`Started ${run.run_id}`);
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useStopRunMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: stopRun,
    onSuccess: (run) => {
      showToast(`Stop requested for ${run.run_id}`);
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useLoadNodeMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ runId, nodeId }: { runId: string; nodeId: string }) => loadRunNode(runId, nodeId),
    onSuccess: (_run, variables) => {
      queryClient.invalidateQueries({ queryKey: ["run-snapshot", variables.runId] });
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useUnloadNodeMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ runId, nodeId }: { runId: string; nodeId: string }) => unloadRunNode(runId, nodeId),
    onSuccess: (_run, variables) => {
      queryClient.invalidateQueries({ queryKey: ["run-snapshot", variables.runId] });
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}
