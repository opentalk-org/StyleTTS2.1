import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { deleteWorkflow, fetchExampleWorkflows, fetchRunGraph, fetchRuns, fetchRunSnapshot, fetchSavedWorkflows, fetchWorkflowSchema, loadRunNode, saveWorkflow, startGraph, stopRun, unloadRunNode } from "./api";

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

// staleTime 0 (default): the folder is re-read every time the library opens, so
// example edits show up without a backend restart.
export function useExampleWorkflowsQuery() {
  return useQuery({ queryKey: ["workflow-examples"], queryFn: fetchExampleWorkflows });
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

export function useDeleteWorkflowMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id }: { id: string; name: string }) => deleteWorkflow(id),
    onSuccess: (_result, variables) => {
      showToast(`Deleted "${variables.name}"`, undefined, "error");
      queryClient.invalidateQueries({ queryKey: ["workflows"] });
    },
    onError: (error, variables) => {
      showToast(`Couldn't delete "${variables.name}"`, error instanceof Error ? error.message : undefined, "error");
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
